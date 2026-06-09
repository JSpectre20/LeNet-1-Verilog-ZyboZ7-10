`timescale 1ns/1ps

module lenet1_top #(
    parameter CONV1_WEIGHTS_FILE = "conv2d_1_W.mem",
    parameter CONV1_BIAS_FILE = "conv2d_1_B.mem",
    parameter CONV2_WEIGHTS_FILE = "conv2d_2_W.mem",
    parameter CONV2_BIAS_FILE = "conv2d_2_B.mem",
    parameter DENSE_WEIGHTS_FILE = "dense_1_W.mem",
    parameter DENSE_BIAS_FILE = "dense_1_B.mem"
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire signed [7:0]        in_data,

    output wire                     out_valid,
    input  wire                     out_ready,
    output wire [3:0]               out_class,
    output wire signed [63:0]       out_score,
    output wire                     frame_done
);
    wire conv1_valid;
    wire conv1_ready;
    wire signed [31:0] conv1_data;
    wire [2:0] conv1_ch;
    wire [4:0] conv1_x;
    wire [4:0] conv1_y;

    wire pool1_valid;
    wire pool1_ready;
    wire signed [31:0] pool1_data;
    wire [2:0] pool1_ch;
    wire [3:0] pool1_x;
    wire [3:0] pool1_y;

    wire conv2_valid;
    wire conv2_ready;
    wire signed [47:0] conv2_data;
    wire [3:0] conv2_ch;
    wire [2:0] conv2_x;
    wire [2:0] conv2_y;

    wire pool2_valid;
    wire pool2_ready;
    wire signed [47:0] pool2_data;
    wire [3:0] pool2_ch;
    wire [1:0] pool2_x;
    wire [1:0] pool2_y;

    wire dense_valid;
    wire dense_ready;
    wire signed [63:0] dense_data;
    wire [3:0] dense_ch;

    conv1 #(
        .DATA_W(8),
        .WEIGHT_W(16),
        .ACC_W(32),
        .IN_W(28),
        .IN_H(28),
        .K(5),
        .OUT_CH(6),
        .CH_W(3),
        .XW(5),
        .YW(5),
        .WEIGHTS_FILE(CONV1_WEIGHTS_FILE),
        .BIAS_FILE(CONV1_BIAS_FILE)
    ) u_conv1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .out_valid(conv1_valid),
        .out_ready(conv1_ready),
        .out_data(conv1_data),
        .out_ch(conv1_ch),
        .out_x(conv1_x),
        .out_y(conv1_y),
        .frame_done()
    );

    avgpool1 #(
        .DATA_W(32),
        .IN_W(24),
        .IN_H(24),
        .CH(6),
        .CH_W(3),
        .IN_XW(5),
        .IN_YW(5),
        .OUT_XW(4),
        .OUT_YW(4)
    ) u_avgpool1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(conv1_valid),
        .in_ready(conv1_ready),
        .in_data(conv1_data),
        .in_ch(conv1_ch),
        .in_x(conv1_x),
        .in_y(conv1_y),
        .out_valid(pool1_valid),
        .out_ready(pool1_ready),
        .out_data(pool1_data),
        .out_ch(pool1_ch),
        .out_x(pool1_x),
        .out_y(pool1_y),
        .frame_done()
    );

    conv2 #(
        .DATA_W(32),
        .WEIGHT_W(16),
        .ACC_W(48),
        .IN_W(12),
        .IN_H(12),
        .IN_CH(6),
        .OUT_CH(12),
        .K(5),
        .IN_CH_W(3),
        .OUT_CH_W(4),
        .IN_XW(4),
        .IN_YW(4),
        .OUT_XW(3),
        .OUT_YW(3),
        .WEIGHTS_FILE(CONV2_WEIGHTS_FILE),
        .BIAS_FILE(CONV2_BIAS_FILE)
    ) u_conv2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pool1_valid),
        .in_ready(pool1_ready),
        .in_data(pool1_data),
        .in_ch(pool1_ch),
        .in_x(pool1_x),
        .in_y(pool1_y),
        .out_valid(conv2_valid),
        .out_ready(conv2_ready),
        .out_data(conv2_data),
        .out_ch(conv2_ch),
        .out_x(conv2_x),
        .out_y(conv2_y),
        .frame_done()
    );

    avgpool2 #(
        .DATA_W(48),
        .IN_W(8),
        .IN_H(8),
        .CH(12),
        .CH_W(4),
        .IN_XW(3),
        .IN_YW(3),
        .OUT_XW(2),
        .OUT_YW(2)
    ) u_avgpool2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(conv2_valid),
        .in_ready(conv2_ready),
        .in_data(conv2_data),
        .in_ch(conv2_ch),
        .in_x(conv2_x),
        .in_y(conv2_y),
        .out_valid(pool2_valid),
        .out_ready(pool2_ready),
        .out_data(pool2_data),
        .out_ch(pool2_ch),
        .out_x(pool2_x),
        .out_y(pool2_y),
        .frame_done()
    );

    dense #(
        .DATA_W(48),
        .WEIGHT_W(16),
        .ACC_W(64),
        .IN_W(4),
        .IN_H(4),
        .IN_CH(12),
        .OUT_CH(10),
        .IN_CH_W(4),
        .IN_XW(2),
        .IN_YW(2),
        .OUT_CH_W(4),
        .WEIGHTS_FILE(DENSE_WEIGHTS_FILE),
        .BIAS_FILE(DENSE_BIAS_FILE)
    ) u_dense (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pool2_valid),
        .in_ready(pool2_ready),
        .in_data(pool2_data),
        .in_ch(pool2_ch),
        .in_x(pool2_x),
        .in_y(pool2_y),
        .out_valid(dense_valid),
        .out_ready(dense_ready),
        .out_data(dense_data),
        .out_ch(dense_ch),
        .frame_done()
    );

    argmax #(
        .DATA_W(64),
        .CLASS_COUNT(10),
        .CLASS_W(4)
    ) u_argmax (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(dense_valid),
        .in_ready(dense_ready),
        .in_data(dense_data),
        .in_ch(dense_ch),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_class(out_class),
        .out_score(out_score),
        .frame_done(frame_done)
    );
endmodule
