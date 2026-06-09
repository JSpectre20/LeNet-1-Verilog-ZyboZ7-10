`timescale 1ns/1ps

// dense - BRAM-backed version (v3 : FIXED-POINT CORRECTED).
// ---------------------------------------------------------------------------
// Changes vs v2 (math only - FSM / ports / BRAM structure unchanged):
//   * Inputs Q5.10, weights Q5.10 -> products Q.20.  Accumulate products only
//     (accum seeded 0, NOT bias), then at emit:
//          logit_q510 = ( SUM(in * Wq) >>> 10 ) + Bq
//     No ReLU here - this is the final classification layer (softmax/logits),
//     and argmax is scale-invariant.
//   * v2 seeded accum with the bias (Q5.10) and added Q.20 products, so the bias
//     was 2^20 too small and there was no rescale.  Both fixed here.
module dense #(
    parameter integer DATA_W = 48,
    parameter integer WEIGHT_W = 8,
    parameter integer ACC_W = 64,
    parameter integer IN_W = 4,
    parameter integer IN_H = 4,
    parameter integer IN_CH = 12,
    parameter integer OUT_CH = 10,
    parameter integer IN_CH_W = 4,
    parameter integer IN_XW = 2,
    parameter integer IN_YW = 2,
    parameter integer OUT_CH_W = 4,
    parameter integer FRAC = 10,            // undo doubled fractional bits
    parameter WEIGHTS_FILE = "dense_1_W.mem",
    parameter BIAS_FILE = "dense_1_B.mem"
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
    output wire                         frame_done
);
    localparam integer VECTOR_LEN   = IN_W * IN_H * IN_CH;
    localparam integer WEIGHT_COUNT = OUT_CH * VECTOR_LEN;
    localparam integer PROD_W       = DATA_W + WEIGHT_W;

    localparam [1:0] S_LOAD = 2'd0;
    localparam [1:0] S_MAC  = 2'd1;
    localparam [1:0] S_OUT  = 2'd2;

    reg [1:0] state;

    // ---- block-memory arrays (driven only in the synchronous blocks below) ----
    (* ram_style = "block" *) reg signed [DATA_W-1:0]   vector_mem [0:VECTOR_LEN-1];
    (* rom_style = "block" *) reg signed [WEIGHT_W-1:0]  weight_mem [0:WEIGHT_COUNT-1];
    reg signed [ACC_W-1:0] bias_mem [0:OUT_CH-1];   // tiny, stays in fabric

    reg [15:0]             load_count;
    reg [15:0]             mac_idx;     // 0..VECTOR_LEN
    reg [OUT_CH_W-1:0]     cur_out;
    reg signed [ACC_W-1:0] accum;

    // BRAM output registers (no reset -> loaded before use)
    reg signed [DATA_W-1:0]   v_q;
    reg signed [WEIGHT_W-1:0] w_q;

    reg                    out_valid_r;
    reg signed [ACC_W-1:0] out_data_r;
    reg [OUT_CH_W-1:0]     out_ch_r;

    wire input_fire  = in_valid && in_ready;
    wire output_fire = out_valid_r && out_ready;
    wire [15:0] vector_index  = in_ch*IN_W*IN_H + in_y*IN_W + in_x;
    wire [31:0] weight_index  = cur_out*VECTOR_LEN + mac_idx;

    wire signed [PROD_W-1:0] product     = v_q * w_q;
    wire signed [ACC_W-1:0]  product_ext =
        {{(ACC_W-PROD_W){product[PROD_W-1]}}, product};
    wire signed [ACC_W-1:0]  mac_sum     = accum + product_ext;

    // rescale (Q.20 -> Q5.10) -> add bias (Q5.10).  No ReLU (final logits).
    wire signed [ACC_W-1:0]  fc_logit = (mac_sum >>> FRAC) + bias_mem[cur_out];

    assign in_ready   = (state == S_LOAD);
    assign out_valid  = out_valid_r;
    assign out_data   = out_data_r;
    assign out_ch     = out_ch_r;
    assign frame_done = output_fire && (out_ch_r == OUT_CH-1);

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
    //  Dedicated synchronous memory block  ->  infers block RAM / ROM
    // =========================================================================
    always @(posedge clk) begin
        // vector_mem : write during load, registered read during MAC (SDP RAM)
        if (input_fire)
            vector_mem[vector_index] <= in_data;
        if ((state == S_MAC) && (mac_idx <= VECTOR_LEN-1)) begin
            v_q <= vector_mem[mac_idx];
            w_q <= weight_mem[weight_index];
        end
    end

    // =========================================================================
    //  Control / datapath FSM  (async reset, no memory arrays touched here)
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_LOAD;
            load_count  <= 16'd0;
            mac_idx     <= 16'd0;
            cur_out     <= {OUT_CH_W{1'b0}};
            accum       <= {ACC_W{1'b0}};
            out_valid_r <= 1'b0;
            out_data_r  <= {ACC_W{1'b0}};
            out_ch_r    <= {OUT_CH_W{1'b0}};
        end else begin
            if (output_fire)
                out_valid_r <= 1'b0;

            // ---------- load the 192-element feature vector ----------
            if (input_fire) begin
                if (load_count == VECTOR_LEN-1) begin
                    load_count <= 16'd0;
                    cur_out    <= {OUT_CH_W{1'b0}};
                    mac_idx    <= 16'd0;
                    accum      <= {ACC_W{1'b0}};    // products only
                    state      <= S_MAC;
                end else begin
                    load_count <= load_count + 16'd1;
                end
            end

            // ---------- serial MAC (1-cycle BRAM read latency) ----------
            if (state == S_MAC) begin
                if (mac_idx >= 16'd1) begin
                    if (mac_idx == VECTOR_LEN) begin
                        out_valid_r <= 1'b1;
                        out_data_r  <= fc_logit;    // rescale + bias
                        out_ch_r    <= cur_out;
                        state       <= S_OUT;
                    end else begin
                        accum <= mac_sum;
                    end
                end
                if (mac_idx != VECTOR_LEN)
                    mac_idx <= mac_idx + 16'd1;
            end

            if (state == S_OUT && output_fire) begin
                out_valid_r <= 1'b0;
                if (cur_out == OUT_CH-1) begin
                    state   <= S_LOAD;
                    cur_out <= {OUT_CH_W{1'b0}};
                    mac_idx <= 16'd0;
                    accum   <= {ACC_W{1'b0}};
                end else begin
                    cur_out <= cur_out + {{(OUT_CH_W-1){1'b0}}, 1'b1};
                    mac_idx <= 16'd0;
                    accum   <= {ACC_W{1'b0}};       // products only
                    state   <= S_MAC;
                end
            end
        end
    end
endmodule
