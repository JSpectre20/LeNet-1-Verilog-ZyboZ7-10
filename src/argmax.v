`timescale 1ns/1ps

module argmax #(
    parameter integer DATA_W = 64,
    parameter integer CLASS_COUNT = 10,
    parameter integer CLASS_W = 4
) (
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         in_valid,
    output wire                         in_ready,
    input  wire signed [DATA_W-1:0]     in_data,
    input  wire [CLASS_W-1:0]           in_ch,

    output wire                         out_valid,
    input  wire                         out_ready,
    output wire [CLASS_W-1:0]           out_class,
    output wire signed [DATA_W-1:0]     out_score,
    output wire                         frame_done
);
    reg [CLASS_W-1:0] count;
    reg [CLASS_W-1:0] best_class;
    reg signed [DATA_W-1:0] best_score;

    reg out_valid_r;
    reg [CLASS_W-1:0] out_class_r;
    reg signed [DATA_W-1:0] out_score_r;

    wire input_fire = in_valid && in_ready;
    wire output_fire = out_valid_r && out_ready;
    wire first_logit = (count == {CLASS_W{1'b0}});
    wire better = first_logit || (in_data > best_score);
    wire last_logit = (count == CLASS_COUNT-1);

    assign in_ready = !out_valid_r || out_ready;
    assign out_valid = out_valid_r;
    assign out_class = out_class_r;
    assign out_score = out_score_r;
    assign frame_done = output_fire;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= {CLASS_W{1'b0}};
            best_class <= {CLASS_W{1'b0}};
            best_score <= {DATA_W{1'b0}};
            out_valid_r <= 1'b0;
            out_class_r <= {CLASS_W{1'b0}};
            out_score_r <= {DATA_W{1'b0}};
        end else begin
            if (output_fire)
                out_valid_r <= 1'b0;

            if (input_fire) begin
                if (last_logit) begin
                    out_valid_r <= 1'b1;
                    out_class_r <= better ? in_ch : best_class;
                    out_score_r <= better ? in_data : best_score;
                    count <= {CLASS_W{1'b0}};
                    best_class <= {CLASS_W{1'b0}};
                    best_score <= {DATA_W{1'b0}};
                end else begin
                    count <= count + {{(CLASS_W-1){1'b0}}, 1'b1};
                    if (better) begin
                        best_class <= in_ch;
                        best_score <= in_data;
                    end
                end
            end
        end
    end
endmodule
