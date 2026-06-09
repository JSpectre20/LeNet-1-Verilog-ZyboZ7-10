`timescale 1ns/1ps

// conv2 - BRAM frame-buffer version (v4 : FIXED-POINT CORRECTED).
// ---------------------------------------------------------------------------
// Changes vs v3 (math only - FSM / ports / BRAM structure unchanged):
//   * Inputs are Q5.10, weights are Q5.10 -> products are Q.20.  We accumulate
//     products only (acc seeded with 0, NOT bias), then at emit:
//          out_q510 = ( SUM(in * Wq) >>> 10 ) + Bq      // back to Q5.10
//     then apply ReLU (LeNet-1 conv layers use relu).
//   * v3 seeded acc with the bias (Q5.10) and added Q.20 products to it, so the
//     bias was 2^10 too small and there was no rescale -> fractional bits piled
//     up and the bias was effectively ignored.  Both are fixed here.
module conv2 #(
    parameter integer DATA_W = 32,
    parameter integer WEIGHT_W = 8,
    parameter integer ACC_W = 48,
    parameter integer IN_W = 12,
    parameter integer IN_H = 12,
    parameter integer IN_CH = 6,
    parameter integer OUT_CH = 12,
    parameter integer K = 5,
    parameter integer IN_CH_W = 3,
    parameter integer OUT_CH_W = 4,
    parameter integer IN_XW = 4,
    parameter integer IN_YW = 4,
    parameter integer OUT_XW = 3,
    parameter integer OUT_YW = 3,
    parameter integer FRAC = 10,            // undo doubled fractional bits
    parameter WEIGHTS_FILE = "conv2d_2_W.mem",
    parameter BIAS_FILE = "conv2d_2_B.mem"
) (
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire signed [DATA_W-1:0]     in_data,
    input  wire [IN_CH_W-1:0]           in_ch,
    input  wire [IN_XW-1:0]             in_x,
    input  wire [IN_YW-1:0]             in_y,

    output wire                         out_valid,
    input  wire                         out_ready,
    output wire signed [ACC_W-1:0]      out_data,
    output wire [OUT_CH_W-1:0]          out_ch,
    output wire [OUT_XW-1:0]            out_x,
    output wire [OUT_YW-1:0]            out_y,
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

    localparam integer OUT_W        = IN_W - K + 1;
    localparam integer OUT_H        = IN_H - K + 1;
    localparam integer TAP_COUNT    = K * K;
    localparam integer WEIGHT_COUNT = OUT_CH * IN_CH * TAP_COUNT;
    localparam integer FRAME_SIZE   = IN_CH * IN_H * IN_W;     // 864
    localparam integer TW           = clog2(K);
    localparam integer PROD_W       = DATA_W + WEIGHT_W;

    // ---- block-memory arrays (driven only in the synchronous blocks) ----
    (* ram_style = "block" *) reg signed [DATA_W-1:0]   frame_mem  [0:FRAME_SIZE-1];
    (* rom_style = "block" *) reg signed [WEIGHT_W-1:0]  weight_mem [0:WEIGHT_COUNT-1];
    reg signed [ACC_W-1:0] bias_mem [0:OUT_CH-1];   // tiny, stays in fabric

    // ---- FSM ----
    localparam [2:0] S_LOAD  = 3'd0,
                     S_FETCH = 3'd1,
                     S_EXEC  = 3'd2,
                     S_EMIT  = 3'd3,
                     S_DONE  = 3'd4;
    reg [2:0]              state;

    reg [15:0]             load_count;
    // output position + channel
    reg [OUT_YW-1:0]       oy;
    reg [OUT_XW-1:0]       ox;
    reg [OUT_CH_W-1:0]     och;
    // tap counters
    reg [IN_CH_W-1:0]      ich;
    reg [TW-1:0]           tr;
    reg [TW-1:0]           tc;

    reg signed [ACC_W-1:0] acc;

    // registered block-RAM read outputs
    reg signed [DATA_W-1:0]   frame_q;
    reg signed [WEIGHT_W-1:0] wgt_q;

    reg                    out_valid_r;
    reg signed [ACC_W-1:0] out_data_r;
    reg [OUT_CH_W-1:0]     out_ch_r;
    reg [OUT_XW-1:0]       out_x_r;
    reg [OUT_YW-1:0]       out_y_r;

    wire input_fire  = in_valid && in_ready;
    wire output_fire = out_valid_r && out_ready;
    wire last_tap    = (ich == IN_CH-1) && (tr == K-1) && (tc == K-1);

    // read addresses (combinational from the counters)
    wire [15:0] frame_waddr = in_ch*IN_H*IN_W + in_y*IN_W + in_x;
    wire [15:0] frame_raddr = ich*IN_H*IN_W + (oy+tr)*IN_W + (ox+tc);
    wire [15:0] weight_addr = och*IN_CH*TAP_COUNT + ich*TAP_COUNT + tr*K + tc;

    wire signed [PROD_W-1:0] product     = frame_q * wgt_q;
    wire signed [ACC_W-1:0]  product_ext =
        {{(ACC_W-PROD_W){product[PROD_W-1]}}, product};
    wire signed [ACC_W-1:0]  acc_next     = acc + product_ext;

    // rescale (Q.20 -> Q5.10) -> add bias (Q5.10) -> ReLU
    wire signed [ACC_W-1:0]  c2_scaled = acc_next >>> FRAC;
    wire signed [ACC_W-1:0]  c2_biased = c2_scaled + bias_mem[och];
    wire signed [ACC_W-1:0]  c2_relu   = c2_biased[ACC_W-1] ? {ACC_W{1'b0}} : c2_biased;

    assign in_ready   = (state == S_LOAD);
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
        for (init_i = 0; init_i < WEIGHT_COUNT; init_i = init_i + 1)
            weight_mem[init_i] = {WEIGHT_W{1'b0}};
        for (init_i = 0; init_i < OUT_CH; init_i = init_i + 1)
            bias_mem[init_i] = {ACC_W{1'b0}};
        if (WEIGHTS_FILE != "") $readmemh(WEIGHTS_FILE, weight_mem);
        if (BIAS_FILE    != "") $readmemh(BIAS_FILE, bias_mem);
    end

    // =========================================================================
    //  Dedicated synchronous block-RAM access  ->  infers BRAM / block ROM
    // =========================================================================
    always @(posedge clk) begin
        if (input_fire)                 // write the frame during load
            frame_mem[frame_waddr] <= in_data;
        if (state == S_FETCH) begin     // registered tap read during MAC
            frame_q <= frame_mem[frame_raddr];
            wgt_q   <= weight_mem[weight_addr];
        end
    end

    // =========================================================================
    //  Control / datapath FSM (async reset; no block-memory arrays touched)
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_LOAD;
            load_count  <= 16'd0;
            oy <= {OUT_YW{1'b0}};  ox <= {OUT_XW{1'b0}};  och <= {OUT_CH_W{1'b0}};
            ich <= {IN_CH_W{1'b0}}; tr <= {TW{1'b0}};      tc  <= {TW{1'b0}};
            acc         <= {ACC_W{1'b0}};
            out_valid_r <= 1'b0;
            out_data_r  <= {ACC_W{1'b0}};
            out_ch_r    <= {OUT_CH_W{1'b0}};
            out_x_r     <= {OUT_XW{1'b0}};
            out_y_r     <= {OUT_YW{1'b0}};
        end else begin
            if (output_fire)
                out_valid_r <= 1'b0;

            case (state)
                // ---------- buffer the whole input frame ----------
                S_LOAD: begin
                    if (input_fire) begin
                        if (load_count == FRAME_SIZE-1) begin
                            load_count <= 16'd0;
                            oy  <= {OUT_YW{1'b0}};  ox <= {OUT_XW{1'b0}};
                            och <= {OUT_CH_W{1'b0}};
                            ich <= {IN_CH_W{1'b0}}; tr <= {TW{1'b0}};
                            tc  <= {TW{1'b0}};
                            acc <= {ACC_W{1'b0}};       // products only
                            state <= S_FETCH;
                        end else begin
                            load_count <= load_count + 16'd1;
                        end
                    end
                end

                // ---------- present tap address (BRAM read happens this edge) ----------
                S_FETCH: begin
                    state <= S_EXEC;
                end

                // ---------- multiply-accumulate the fetched tap ----------
                S_EXEC: begin
                    if (last_tap) begin
                        out_data_r  <= c2_relu;     // rescale + bias + ReLU
                        out_valid_r <= 1'b1;
                        out_ch_r    <= och;
                        out_x_r     <= ox[OUT_XW-1:0];
                        out_y_r     <= oy[OUT_YW-1:0];
                        state       <= S_EMIT;
                    end else begin
                        acc <= acc_next;
                        // advance tap: tc -> tr -> ich
                        if (tc == K-1) begin
                            tc <= {TW{1'b0}};
                            if (tr == K-1) begin
                                tr  <= {TW{1'b0}};
                                ich <= ich + 1'b1;
                            end else begin
                                tr <= tr + 1'b1;
                            end
                        end else begin
                            tc <= tc + 1'b1;
                        end
                        state <= S_FETCH;
                    end
                end

                // ---------- emit one output; advance och / ox / oy ----------
                S_EMIT: begin
                    if (output_fire) begin
                        ich <= {IN_CH_W{1'b0}}; tr <= {TW{1'b0}}; tc <= {TW{1'b0}};
                        acc <= {ACC_W{1'b0}};           // products only
                        if (och == OUT_CH-1) begin
                            och <= {OUT_CH_W{1'b0}};
                            if (ox == OUT_W-1) begin
                                ox <= {OUT_XW{1'b0}};
                                if (oy == OUT_H-1) begin
                                    state <= S_DONE;      // whole frame done
                                end else begin
                                    oy    <= oy + 1'b1;
                                    state <= S_FETCH;
                                end
                            end else begin
                                ox    <= ox + 1'b1;
                                state <= S_FETCH;
                            end
                        end else begin
                            och   <= och + 1'b1;
                            state <= S_FETCH;
                        end
                    end
                end

                // ---------- one-shot inference complete ----------
                S_DONE: begin
                    state <= S_DONE;   // stay here (single frame)
                end

                default: state <= S_LOAD;
            endcase
        end
    end
endmodule
