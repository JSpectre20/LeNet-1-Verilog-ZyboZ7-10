`timescale 1ns/1ps

// conv1 - serialized + BRAM-weights version (v3 : FIXED-POINT CORRECTED).
// ---------------------------------------------------------------------------
// Changes vs v2 (math only - FSM / ports / resources unchanged):
//   1. Pixel is now read UNSIGNED (0..255).  v2 fed it through a signed[7:0]
//      port, so any pixel >= 0x80 became negative.  We zero-extend before the
//      multiply: $signed({1'b0, win_val}).
//   2. Input normalisation x = pixel/256 (~= pixel/255) is folded into the
//      output rescale.  weights/biases are Q5.10, so:
//          conv_out_q510 = ( SUM(pixel * Wq) >>> 8 ) + Bq
//      The >>> 8 = (1/256) * (1/2^10 to undo the doubled fractional bits) and
//      lands the result back in Q5.10, the SAME format every other layer uses.
//   3. Bias is added AFTER the rescale (at Q5.10), not seeded into acc.
//   4. ReLU is applied to the final value (LeNet-1 conv layers use relu).
// acc therefore accumulates ONLY products now; bias seeding is removed.
module conv1 #(
    parameter integer DATA_W = 8,
    parameter integer WEIGHT_W = 8,
    parameter integer ACC_W = 32,
    parameter integer IN_W = 28,
    parameter integer IN_H = 28,
    parameter integer K = 5,
    parameter integer OUT_CH = 6,
    parameter integer CH_W = 3,
    parameter integer XW = 5,
    parameter integer YW = 5,
    parameter integer IN_FRAC = 8,          // input shift: pixel/256 + Q5.10 align
    parameter WEIGHTS_FILE = "conv2d_1_W.mem",
    parameter BIAS_FILE = "conv2d_1_B.mem"
) (
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire signed [DATA_W-1:0]     in_data,

    output wire                         out_valid,
    input  wire                         out_ready,
    output wire signed [ACC_W-1:0]      out_data,
    output wire [CH_W-1:0]              out_ch,
    output wire [XW-1:0]                out_x,
    output wire [YW-1:0]                out_y,
    output wire                         frame_done
);
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    localparam integer OUT_W     = IN_W - K + 1;
    localparam integer OUT_H     = IN_H - K + 1;
    localparam integer IN_XW     = clog2(IN_W);
    localparam integer IN_YW     = clog2(IN_H);
    localparam integer TAP_COUNT = K * K;
    localparam integer TW        = clog2(K);

    // ---- weights : registered read -> block ROM ; bias : tiny, fabric ----
    (* rom_style = "block" *) reg signed [WEIGHT_W-1:0] weight_mem [0:OUT_CH*TAP_COUNT-1];
    reg signed [ACC_W-1:0] bias_mem [0:OUT_CH-1];

    // ---- line-buffer + sliding window front-end (fabric shift registers) ----
    reg signed [DATA_W-1:0] row0 [0:IN_W-1];
    reg signed [DATA_W-1:0] row1 [0:IN_W-1];
    reg signed [DATA_W-1:0] row2 [0:IN_W-1];
    reg signed [DATA_W-1:0] row3 [0:IN_W-1];
    reg signed [DATA_W-1:0] window [0:K-1][0:K-1];

    reg [IN_XW-1:0] in_x;
    reg [IN_YW-1:0] in_y;

    // ---- serial MAC FSM (2 cycles / tap to absorb the ROM read latency) ----
    localparam [1:0] S_IDLE = 2'd0, S_FETCH = 2'd1, S_EXEC = 2'd2, S_WAIT = 2'd3;
    reg [1:0]              state;
    reg [CH_W-1:0]         mac_ch;
    reg [TW-1:0]           tap_r;
    reg [TW-1:0]           tap_c;
    reg signed [ACC_W-1:0] acc;
    reg [XW-1:0]           hold_x;
    reg [YW-1:0]           hold_y;
    reg signed [WEIGHT_W-1:0] wgt_q;       // registered weight read (no reset)

    reg                    out_valid_r;
    reg signed [ACC_W-1:0] out_data_r;
    reg [CH_W-1:0]         out_ch_r;
    reg [XW-1:0]           out_x_r;
    reg [YW-1:0]           out_y_r;

    wire input_fire  = in_valid && in_ready;
    wire output_fire = out_valid_r && out_ready;
    wire last_tap    = (tap_r == K-1) && (tap_c == K-1);
    wire [31:0] weight_addr = mac_ch*TAP_COUNT + tap_r*K + tap_c;

    assign in_ready   = (state == S_IDLE);
    assign out_valid  = out_valid_r;
    assign out_data   = out_data_r;
    assign out_ch     = out_ch_r;
    assign out_x      = out_x_r;
    assign out_y      = out_y_r;
    assign frame_done = output_fire &&
                        (out_ch_r == OUT_CH-1) &&
                        (out_x_r == OUT_W-1) &&
                        (out_y_r == OUT_H-1);

    integer init_i;
    initial begin
        for (init_i = 0; init_i < OUT_CH*TAP_COUNT; init_i = init_i + 1)
            weight_mem[init_i] = {WEIGHT_W{1'b0}};
        for (init_i = 0; init_i < OUT_CH; init_i = init_i + 1)
            bias_mem[init_i] = {ACC_W{1'b0}};
        if (WEIGHTS_FILE != "") $readmemh(WEIGHTS_FILE, weight_mem);
        if (BIAS_FILE    != "") $readmemh(BIAS_FILE, bias_mem);
    end

    // =========================================================================
    //  Dedicated synchronous weight-ROM read  ->  infers block ROM
    // =========================================================================
    always @(posedge clk) begin
        if (state == S_FETCH)
            wgt_q <= weight_mem[weight_addr];
    end

    integer win_c;
    reg signed [DATA_W-1:0]            t0, t1, t2, t3;
    reg signed [DATA_W-1:0]            win_val;
    reg signed [DATA_W+WEIGHT_W:0]     product;   // unsigned pixel(<=8b) * signed wgt
    // rescale / bias / relu temporaries (combinational, blocking-assigned)
    reg signed [ACC_W-1:0]             c1_full;
    reg signed [ACC_W-1:0]             c1_scaled;
    reg signed [ACC_W-1:0]             c1_biased;

    // =========================================================================
    //  Front-end + MAC control FSM (async reset; no weight_mem access here)
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_x   <= {IN_XW{1'b0}};
            in_y   <= {IN_YW{1'b0}};
            state  <= S_IDLE;
            mac_ch <= {CH_W{1'b0}};
            tap_r  <= {TW{1'b0}};
            tap_c  <= {TW{1'b0}};
            acc    <= {ACC_W{1'b0}};
            hold_x <= {XW{1'b0}};
            hold_y <= {YW{1'b0}};
            out_valid_r <= 1'b0;
            out_data_r  <= {ACC_W{1'b0}};
            out_ch_r    <= {CH_W{1'b0}};
            out_x_r     <= {XW{1'b0}};
            out_y_r     <= {YW{1'b0}};
            // row buffers / window NOT reset (written before read)
        end else begin
            if (output_fire)
                out_valid_r <= 1'b0;

            // ---------- front-end: accept one pixel when idle ----------
            if (input_fire) begin
                t0 = row0[in_x];
                t1 = row1[in_x];
                t2 = row2[in_x];
                t3 = row3[in_x];

                row0[in_x] <= in_data;
                row1[in_x] <= t0;
                row2[in_x] <= t1;
                row3[in_x] <= t2;

                for (win_c = 0; win_c < K-1; win_c = win_c + 1) begin
                    window[0][win_c] <= window[0][win_c+1];
                    window[1][win_c] <= window[1][win_c+1];
                    window[2][win_c] <= window[2][win_c+1];
                    window[3][win_c] <= window[3][win_c+1];
                    window[4][win_c] <= window[4][win_c+1];
                end
                window[0][K-1] <= t3;
                window[1][K-1] <= t2;
                window[2][K-1] <= t1;
                window[3][K-1] <= t0;
                window[4][K-1] <= in_data;

                if ((in_x >= K-1) && (in_y >= K-1)) begin
                    state  <= S_FETCH;
                    mac_ch <= {CH_W{1'b0}};
                    tap_r  <= {TW{1'b0}};
                    tap_c  <= {TW{1'b0}};
                    acc    <= {ACC_W{1'b0}};        // accumulate products only
                    hold_x <= in_x - (K-1);
                    hold_y <= in_y - (K-1);
                end

                if (in_x == IN_W-1) begin
                    in_x <= {IN_XW{1'b0}};
                    if (in_y == IN_H-1) in_y <= {IN_YW{1'b0}};
                    else                in_y <= in_y + 1'b1;
                end else begin
                    in_x <= in_x + 1'b1;
                end
            end

            // ---------- serial multiply-accumulate ----------
            case (state)
                S_FETCH: begin
                    state <= S_EXEC;   // wgt_q loaded by the memory block this edge
                end

                S_EXEC: begin
                    win_val = window[tap_r][tap_c];
                    // pixel is UNSIGNED 0..255 -> zero-extend to keep it positive
                    product = $signed({1'b0, win_val}) * wgt_q;

                    if (last_tap) begin
                        // rescale to Q5.10  ->  add bias (Q5.10)  ->  ReLU
                        c1_full   = acc + product;
                        c1_scaled = c1_full >>> IN_FRAC;
                        c1_biased = c1_scaled + bias_mem[mac_ch];
                        out_data_r  <= c1_biased[ACC_W-1] ? {ACC_W{1'b0}} : c1_biased;
                        out_valid_r <= 1'b1;
                        out_ch_r    <= mac_ch;
                        out_x_r     <= hold_x;
                        out_y_r     <= hold_y;
                        state       <= S_WAIT;
                    end else begin
                        acc <= acc + product;
                        if (tap_c == K-1) begin
                            tap_c <= {TW{1'b0}};
                            tap_r <= tap_r + 1'b1;
                        end else begin
                            tap_c <= tap_c + 1'b1;
                        end
                        state <= S_FETCH;
                    end
                end

                S_WAIT: begin
                    if (output_fire) begin
                        if (mac_ch == OUT_CH-1) begin
                            state <= S_IDLE;
                        end else begin
                            mac_ch <= mac_ch + 1'b1;
                            tap_r  <= {TW{1'b0}};
                            tap_c  <= {TW{1'b0}};
                            acc    <= {ACC_W{1'b0}};   // accumulate products only
                            state  <= S_FETCH;
                        end
                    end
                end

                default: ; // S_IDLE handled by front-end above
            endcase
        end
    end
endmodule