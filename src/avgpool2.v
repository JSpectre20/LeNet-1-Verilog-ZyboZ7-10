`timescale 1ns/1ps

module avgpool2 #(
    parameter integer DATA_W = 48,
    parameter integer IN_W = 8,
    parameter integer IN_H = 8,
    parameter integer CH = 12,
    parameter integer CH_W = 4,
    parameter integer IN_XW = 3,
    parameter integer IN_YW = 3,
    parameter integer OUT_XW = 2,
    parameter integer OUT_YW = 2
) (
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire signed [DATA_W-1:0]     in_data,
    input  wire [CH_W-1:0]              in_ch,
    input  wire [IN_XW-1:0]             in_x,
    input  wire [IN_YW-1:0]             in_y,

    output wire                         out_valid,
    input  wire                         out_ready,
    output wire signed [DATA_W-1:0]     out_data,
    output wire [CH_W-1:0]              out_ch,
    output wire [OUT_XW-1:0]            out_x,
    output wire [OUT_YW-1:0]            out_y,
    output wire                         frame_done
);
    localparam integer OUT_W = IN_W / 2;
    localparam integer OUT_H = IN_H / 2;
    localparam integer SUM_W = DATA_W + 2;

    reg signed [DATA_W-1:0] left_sample [0:CH-1];
    reg signed [SUM_W-1:0] top_pair_sum [0:CH*OUT_W-1];

    reg out_valid_r;
    reg signed [DATA_W-1:0] out_data_r;
    reg [CH_W-1:0] out_ch_r;
    reg [OUT_XW-1:0] out_x_r;
    reg [OUT_YW-1:0] out_y_r;

    wire output_slot_free = !out_valid_r || out_ready;
    wire input_fire = in_valid && in_ready;
    wire x_is_odd = in_x[0];
    wire y_is_odd = in_y[0];
    wire will_emit = input_fire && x_is_odd && y_is_odd;

    wire [OUT_XW-1:0] pool_x = in_x[IN_XW-1:1];
    wire [OUT_YW-1:0] pool_y = in_y[IN_YW-1:1];
    wire signed [SUM_W-1:0] data_ext = {{2{in_data[DATA_W-1]}}, in_data};
    wire signed [SUM_W-1:0] left_ext =
        {{2{left_sample[in_ch][DATA_W-1]}}, left_sample[in_ch]};
    wire signed [SUM_W-1:0] bottom_pair = left_ext + data_ext;
    wire signed [SUM_W-1:0] pool_sum =
        top_pair_sum[in_ch*OUT_W + pool_x] + bottom_pair;

    assign in_ready = output_slot_free;
    assign out_valid = out_valid_r;
    assign out_data = out_data_r;
    assign out_ch = out_ch_r;
    assign out_x = out_x_r;
    assign out_y = out_y_r;
    assign frame_done = out_valid_r && out_ready &&
                        (out_ch_r == CH-1) &&
                        (out_x_r == OUT_W-1) &&
                        (out_y_r == OUT_H-1);

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_r <= 1'b0;
            out_data_r <= {DATA_W{1'b0}};
            out_ch_r <= {CH_W{1'b0}};
            out_x_r <= {OUT_XW{1'b0}};
            out_y_r <= {OUT_YW{1'b0}};

            // left_sample and top_pair_sum are not reset:
            // left_sample[ch] is always written by the even-x pixel before the
            // odd-x pixel reads it; top_pair_sum[ch][x] is written by the even-y
            // row before the odd-y row reads it. Both are safe without reset.
        end else begin
            if (out_valid_r && out_ready)
                out_valid_r <= 1'b0;

            if (input_fire) begin
                if (!x_is_odd) begin
                    left_sample[in_ch] <= in_data;
                end else if (!y_is_odd) begin
                    top_pair_sum[in_ch*OUT_W + pool_x] <= bottom_pair;
                end

                if (will_emit) begin
                    out_valid_r <= 1'b1;
                    out_data_r <= pool_sum >>> 2;
                    out_ch_r <= in_ch;
                    out_x_r <= pool_x;
                    out_y_r <= pool_y;
                end
            end
        end
    end
endmodule
