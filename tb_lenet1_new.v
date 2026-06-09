// =============================================================================
// tb_lenet1_top.v  -  LeNet-1 testbench (FIXED-POINT CORRECTED golden model)
//
// What changed vs the old TB
// --------------------------
// The previous golden model re-implemented the SAME broken math as the buggy
// RTL (no ReLU, no fractional-bit rescale, signed pixels), so "PASS" only meant
// "DUT matches my broken reference".  This version:
//   * reads pixels UNSIGNED (0..255),
//   * folds input normalisation (pixel/256) + Q5.10 alignment into >>> 8 at
//     conv1, and >>> 10 at conv2 / dense (undo the doubled fractional bits),
//   * adds ReLU after conv1 and conv2,
//   * adds the bias at Q5.10 scale AFTER the rescale,
// so the golden model now matches what LeNet-1 was actually trained to compute.
//
// It also checks the DUT class against EXPECTED_CLASS (the true digit of the
// image), so a PASS now means the hardware is FUNCTIONALLY correct, not just
// self-consistent.  Run each Ik_image.mem with EXPECTED_CLASS=k.
//
// NOTE: the corrected math differs from float Keras only by (a) pixel/256 vs
// /255 (~0.4%) and (b) truncating right-shifts.  For a trained MNIST LeNet-1
// these are negligible; still verify argmax against Keras in Python (cell 7,
// using pixel/255.0 UNSIGNED) as the true golden reference.
// =============================================================================

`timescale 1ns/1ps

module tb_lenet1_top;

    // -------------------------------------------------------------------------
    //  Parameters - override on command line with -generic_top, or edit here.
    //  IMPORTANT: keep IMAGE_FILE and EXPECTED_CLASS in sync (Ik -> k).
    // -------------------------------------------------------------------------
    parameter        IMAGE_FILE     = "I7_image.mem";
    parameter integer EXPECTED_CLASS = 7;
    parameter C1_W_FILE     = "conv2d_1_W.mem";
    parameter C1_B_FILE     = "conv2d_1_B.mem";
    parameter C2_W_FILE     = "conv2d_2_W.mem";
    parameter C2_B_FILE     = "conv2d_2_B.mem";
    parameter FC_W_FILE     = "dense_1_W.mem";
    parameter FC_B_FILE     = "dense_1_B.mem";

    // fractional-bit shifts (must match RTL)
    localparam integer C1_FRAC = 8;    // pixel/256 + undo doubled frac bits
    localparam integer C2_FRAC = 10;   // undo doubled frac bits
    localparam integer FC_FRAC = 10;   // undo doubled frac bits

    // -------------------------------------------------------------------------
    //  Architecture constants  (must match RTL parameters)
    // -------------------------------------------------------------------------
    localparam IMG_W   = 28;
    localparam IMG_H   = 28;
    localparam C1_OUT  = 24;   // conv1 spatial output (28-5+1)
    localparam P1_OUT  = 12;   // avgpool1 spatial output
    localparam C2_OUT  =  8;   // conv2 spatial output (12-5+1)
    localparam P2_OUT  =  4;   // avgpool2 spatial output
    localparam CH1     =  6;   // conv1 output channels
    localparam CH2     = 12;   // conv2 output channels
    localparam CLASSES = 10;
    localparam K       =  5;   // kernel size

    // -------------------------------------------------------------------------
    //  DUT signals
    // -------------------------------------------------------------------------
    reg  clk;
    reg  rst_n;

    reg  in_valid;
    wire in_ready;
    reg  signed [7:0] in_data;

    wire            out_valid;
    reg             out_ready;
    wire [3:0]      out_class;
    wire signed [63:0] out_score;
    wire            frame_done;

    // -------------------------------------------------------------------------
    //  Storage for image and every layer's weights / biases
    // -------------------------------------------------------------------------
    reg [7:0]          image        [0:IMG_W*IMG_H-1];   // UNSIGNED pixels 0..255

    // Weight width matches RTL WEIGHT_W = 16
    reg signed [15:0]  c1_w         [0:CH1*K*K-1];
    reg signed [31:0]  c1_b         [0:CH1-1];

    reg signed [15:0]  c2_w         [0:CH2*CH1*K*K-1];
    reg signed [47:0]  c2_b         [0:CH2-1];

    reg signed [15:0]  fc_w         [0:CLASSES*CH2*P2_OUT*P2_OUT-1];
    reg signed [63:0]  fc_b         [0:CLASSES-1];

    // -------------------------------------------------------------------------
    //  Software reference model intermediates (all Q5.10)
    // -------------------------------------------------------------------------
    reg signed [47:0]  ref_c1       [0:CH1*C1_OUT*C1_OUT-1];
    reg signed [47:0]  ref_p1       [0:CH1*P1_OUT*P1_OUT-1];
    reg signed [63:0]  ref_c2       [0:CH2*C2_OUT*C2_OUT-1];
    reg signed [63:0]  ref_p2       [0:CH2*P2_OUT*P2_OUT-1];
    reg signed [63:0]  ref_logits   [0:CLASSES-1];
    reg        [3:0]   ref_class;
    reg signed [63:0]  ref_score;

    // misc loop variables
    integer ch, och, ich, kr, kc, x, y, i, idx;
    integer send_idx, errors;
    reg signed [63:0] sum;

    // =========================================================================
    //  DUT instantiation
    // =========================================================================
    lenet1_top dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .in_data   (in_data),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_class (out_class),
        .out_score (out_score),
        .frame_done(frame_done)
    );

    // =========================================================================
    //  Clock  - 100 MHz (10 ns period)
    // =========================================================================
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // =========================================================================
    //  Load image + weights from files, then run software reference model
    // =========================================================================
    initial begin : load_and_ref
        $readmemh(IMAGE_FILE, image);
        $readmemh(C1_W_FILE, c1_w);
        $readmemh(C1_B_FILE, c1_b);
        $readmemh(C2_W_FILE, c2_w);
        $readmemh(C2_B_FILE, c2_b);
        $readmemh(FC_W_FILE, fc_w);
        $readmemh(FC_B_FILE, fc_b);

        $display("--------------------------------------------------");
        $display("IMAGE FILE : %s   (expected class = %0d)", IMAGE_FILE, EXPECTED_CLASS);
        $display("Weights loaded from mem files.");
        $display("--------------------------------------------------");

        // ----- conv1 :  relu( (SUM(pixel * Wq) >>> 8) + Bq ) -----
        // pixel is UNSIGNED; zero-extend ({1'b0,...}) so signed multiply stays
        // correct for negative weights.
        for (ch = 0; ch < CH1; ch = ch + 1) begin
            for (y = 0; y < C1_OUT; y = y + 1) begin
                for (x = 0; x < C1_OUT; x = x + 1) begin
                    sum = 64'sd0;
                    for (kr = 0; kr < K; kr = kr + 1)
                        for (kc = 0; kc < K; kc = kc + 1)
                            sum = sum +
                                  $signed({1'b0, image[(y+kr)*IMG_W + (x+kc)]}) *
                                  $signed(c1_w[ch*K*K + kr*K + kc]);
                    sum = (sum >>> C1_FRAC) + c1_b[ch];      // rescale + bias (Q5.10)
                    if (sum < 0) sum = 64'sd0;               // ReLU
                    ref_c1[ch*C1_OUT*C1_OUT + y*C1_OUT + x] = sum[47:0];
                end
            end
        end

        // ----- avgpool1  (2x2 average, >> 2) -----
        for (ch = 0; ch < CH1; ch = ch + 1)
            for (y = 0; y < P1_OUT; y = y + 1)
                for (x = 0; x < P1_OUT; x = x + 1) begin
                    sum = ref_c1[ch*C1_OUT*C1_OUT + (2*y  )*C1_OUT + (2*x  )] +
                          ref_c1[ch*C1_OUT*C1_OUT + (2*y  )*C1_OUT + (2*x+1)] +
                          ref_c1[ch*C1_OUT*C1_OUT + (2*y+1)*C1_OUT + (2*x  )] +
                          ref_c1[ch*C1_OUT*C1_OUT + (2*y+1)*C1_OUT + (2*x+1)];
                    ref_p1[ch*P1_OUT*P1_OUT + y*P1_OUT + x] = (sum >>> 2);
                end

        // ----- conv2 :  relu( (SUM(in * Wq) >>> 10) + Bq ) -----
        for (och = 0; och < CH2; och = och + 1)
            for (y = 0; y < C2_OUT; y = y + 1)
                for (x = 0; x < C2_OUT; x = x + 1) begin
                    sum = 64'sd0;
                    for (ich = 0; ich < CH1; ich = ich + 1)
                        for (kr = 0; kr < K; kr = kr + 1)
                            for (kc = 0; kc < K; kc = kc + 1)
                                sum = sum +
                                      $signed(ref_p1[ich*P1_OUT*P1_OUT +
                                                     (y+kr)*P1_OUT + (x+kc)]) *
                                      $signed(c2_w[och*CH1*K*K + ich*K*K + kr*K + kc]);
                    sum = (sum >>> C2_FRAC) + c2_b[och];     // rescale + bias (Q5.10)
                    if (sum < 0) sum = 64'sd0;               // ReLU
                    ref_c2[och*C2_OUT*C2_OUT + y*C2_OUT + x] = sum;
                end

        // ----- avgpool2  (2x2 average, >> 2) -----
        for (ch = 0; ch < CH2; ch = ch + 1)
            for (y = 0; y < P2_OUT; y = y + 1)
                for (x = 0; x < P2_OUT; x = x + 1) begin
                    sum = ref_c2[ch*C2_OUT*C2_OUT + (2*y  )*C2_OUT + (2*x  )] +
                          ref_c2[ch*C2_OUT*C2_OUT + (2*y  )*C2_OUT + (2*x+1)] +
                          ref_c2[ch*C2_OUT*C2_OUT + (2*y+1)*C2_OUT + (2*x  )] +
                          ref_c2[ch*C2_OUT*C2_OUT + (2*y+1)*C2_OUT + (2*x+1)];
                    ref_p2[ch*P2_OUT*P2_OUT + y*P2_OUT + x] = (sum >>> 2);
                end

        // ----- dense :  logit = (SUM(in * Wq) >>> 10) + Bq   (no ReLU) -----
        for (och = 0; och < CLASSES; och = och + 1) begin
            sum = 64'sd0;
            for (idx = 0; idx < CH2*P2_OUT*P2_OUT; idx = idx + 1)
                sum = sum + $signed(ref_p2[idx]) *
                            $signed(fc_w[och*CH2*P2_OUT*P2_OUT + idx]);
            ref_logits[och] = (sum >>> FC_FRAC) + fc_b[och];
        end

        // ----- argmax -----
        ref_class = 4'd0;
        ref_score = ref_logits[0];
        for (i = 1; i < CLASSES; i = i + 1)
            if (ref_logits[i] > ref_score) begin
                ref_class = i[3:0];
                ref_score = ref_logits[i];
            end

        $display("Reference logits (Q5.10):");
        for (i = 0; i < CLASSES; i = i + 1)
            $display("  class[%0d] = %0d", i, ref_logits[i]);
        $display("Reference: predicted class = %0d  (score = %0d)",
                 ref_class, ref_score);
        $display("--------------------------------------------------");
    end

    // =========================================================================
    //  Reset sequence
    // =========================================================================
    initial begin
        rst_n    = 1'b0;
        in_valid = 1'b0;
        in_data  = 8'sd0;
        out_ready = 1'b0;
        send_idx  = 0;
        errors    = 0;
        repeat (8) @(posedge clk);
        rst_n = 1'b1;
    end

    // =========================================================================
    //  Back-pressure pattern on out_ready  (tests flow-control robustness)
    // =========================================================================
    always @(negedge clk) begin
        if (!rst_n) out_ready <= 1'b0;
        else        out_ready <= (($time / 10) % 4) != 2;
    end

    // =========================================================================
    //  Pixel streaming  (pixel 0..783 sent one per accepted cycle)
    // =========================================================================
    initial begin : pixel_streamer
        @(posedge rst_n);
        @(negedge clk);
        in_valid = 1'b1;
        in_data  = image[0];
        while (send_idx < IMG_W*IMG_H) begin
            @(posedge clk);
            if (in_valid && in_ready) begin
                send_idx = send_idx + 1;
                @(negedge clk);
                if (send_idx < IMG_W*IMG_H) begin
                    in_valid = 1'b1;
                    in_data  = image[send_idx];
                end else begin
                    in_valid = 1'b0;
                    in_data  = 8'sd0;
                    $display("[%0t ns] All %0d pixels streamed to DUT.",
                             $time, IMG_W*IMG_H);
                end
            end
        end
    end

    // =========================================================================
    //  Output checker
    // =========================================================================
    always @(posedge clk) begin
        if (rst_n && out_valid && out_ready) begin
            $display("--------------------------------------------------");
            $display("DUT output : class = %0d  score = %0d", out_class, out_score);
            $display("Reference  : class = %0d  score = %0d", ref_class, ref_score);
            $display("Expected   : class = %0d", EXPECTED_CLASS);

            // --- DUT vs golden model (RTL implements intended fixed-point math) ---
            if (out_class !== ref_class) begin
                $display("FAIL  DUT class %0d != reference class %0d",
                         out_class, ref_class);
                errors = errors + 1;
            end else $display("PASS  DUT matches reference model (%0d)", out_class);

            if (out_score !== ref_score) begin
                $display("FAIL  DUT score %0d != reference score %0d",
                         out_score, ref_score);
                errors = errors + 1;
            end else $display("PASS  score matches (%0d)", out_score);

            // --- functional correctness vs the true digit ---
            if (out_class !== EXPECTED_CLASS[3:0]) begin
                $display("FAIL  predicted %0d but image is a %0d",
                         out_class, EXPECTED_CLASS);
                errors = errors + 1;
            end else $display("PASS  classification correct (%0d)", out_class);

            if (!frame_done) begin
                $display("FAIL  frame_done not asserted with final class");
                errors = errors + 1;
            end else $display("PASS  frame_done asserted");

            $display("--------------------------------------------------");
            if (errors == 0) $display("*** OVERALL: PASS ***");
            else             $display("*** OVERALL: FAIL (%0d error(s)) ***", errors);
            $display("--------------------------------------------------");
            #20;
            $finish;
        end
    end

    // =========================================================================
    //  Simulation timeout
    // =========================================================================
    initial begin
        #10_000_000;
        $display("ERROR: simulation timeout - DUT never produced output.");
        $finish;
    end

endmodule