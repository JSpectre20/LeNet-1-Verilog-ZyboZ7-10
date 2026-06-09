// zybo_top.v  (v2)
// Board-level wrapper for lenet1_top targeting Zybo Z7-10 (xc7z010clg400-1).
//
// FIX vs v1: the board oscillator on K17 is 125 MHz, but the design only closes
// timing around ~50-70 MHz.  v1 just declared sysclk as 50 MHz in the XDC, which
// is a fiction - the hardware still ran at 125 MHz and the datapath fails setup.
// This version uses an MMCM to actually generate a 50 MHz core clock from the
// 125 MHz input.  Everything runs in that single 50 MHz domain (no CDC needed).
//
// External pins (match zybo_top.xdc):
//   sysclk   - K17, 125 MHz board clock
//   btn0     - K18, active-HIGH pushbutton, synchronous reset
//   led[3:0] - M14/M15/G14/D18, latched out_class when result is valid

`timescale 1ns / 1ps

module zybo_top #(
    parameter integer IMAGE_PIXELS = 784,            // 28x28
    parameter         IMAGE_FILE   = "I6_image.mem"  // hex, one byte per line
) (
    input  wire        sysclk,   // 125 MHz (K17)
    input  wire        btn0,     // active-HIGH reset (K18)
    output reg  [3:0]  led       // predicted class (M14/M15/G14/D18)
);

    // ------------------------------------------------------------------ //
    //  Clocking: 125 MHz -> MMCM -> 50 MHz core clock                      //
    //  VCO = 125 * 8 / 1 = 1000 MHz (in 600-1200 range);  1000/20 = 50 MHz //
    // ------------------------------------------------------------------ //
    wire clk_fb, clk_fb_bufg;
    wire clk50, clk50_bufg;
    wire mmcm_locked;

    MMCME2_BASE #(
        .BANDWIDTH        ("OPTIMIZED"),
        .CLKIN1_PERIOD    (8.000),     // 125 MHz
        .DIVCLK_DIVIDE    (1),
        .CLKFBOUT_MULT_F  (8.000),     // VCO = 1000 MHz
        .CLKOUT0_DIVIDE_F (20.000),    // 50 MHz
        .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT0_PHASE    (0.0),
        .STARTUP_WAIT     ("FALSE")
    ) u_mmcm (
        .CLKIN1   (sysclk),
        .CLKFBIN  (clk_fb_bufg),
        .CLKFBOUT (clk_fb),
        .CLKOUT0  (clk50),
        .CLKOUT0B (), .CLKOUT1 (), .CLKOUT1B (), .CLKOUT2 (), .CLKOUT2B (),
        .CLKOUT3 (), .CLKOUT3B (), .CLKOUT4 (), .CLKOUT5 (), .CLKOUT6 (),
        .CLKFBOUTB(),
        .LOCKED   (mmcm_locked),
        .PWRDWN   (1'b0),
        .RST      (1'b0)
    );

    BUFG u_fb_buf  (.I(clk_fb), .O(clk_fb_bufg));
    BUFG u_c0_buf  (.I(clk50),  .O(clk50_bufg));

    wire clk = clk50_bufg;   // single core clock

    // ------------------------------------------------------------------ //
    //  Reset: btn0 synced to clk, held until MMCM locks (active-low rst_n) //
    // ------------------------------------------------------------------ //
    reg [1:0] rst_sync = 2'b11;        // reset asserted at power-up
    always @(posedge clk) rst_sync <= {rst_sync[0], btn0};
    wire rst_n = ~rst_sync[1] & mmcm_locked;

    // ------------------------------------------------------------------ //
    //  Image ROM                                                          //
    // ------------------------------------------------------------------ //
    reg signed [7:0] image_rom [0:IMAGE_PIXELS-1];
    initial $readmemh(IMAGE_FILE, image_rom);

    // ------------------------------------------------------------------ //
    //  Pixel-streaming FSM                                                //
    // ------------------------------------------------------------------ //
    localparam PX_W = $clog2(IMAGE_PIXELS + 1);
    reg [PX_W-1:0]   px_idx;
    reg              in_valid;
    wire             in_ready;
    reg signed [7:0] in_data;

    wire             out_valid;
    reg              out_ready;
    wire [3:0]       out_class;
    wire             frame_done;

    localparam S_IDLE = 2'd0, S_SEND = 2'd1, S_DONE = 2'd2;
    reg [1:0] state;

    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            px_idx    <= 0;
            in_valid  <= 1'b0;
            in_data   <= 8'sd0;
            out_ready <= 1'b1;
        end else begin
            case (state)
                S_IDLE: begin
                    px_idx   <= 0;
                    in_data  <= image_rom[0];
                    in_valid <= 1'b1;
                    state    <= S_SEND;
                end
                S_SEND: begin
                    out_ready <= 1'b1;
                    if (in_valid && in_ready) begin
                        if (px_idx == IMAGE_PIXELS - 1) begin
                            in_valid <= 1'b0;
                            state    <= S_DONE;
                        end else begin
                            px_idx  <= px_idx + 1;
                            in_data <= image_rom[px_idx + 1];
                        end
                    end
                end
                S_DONE: begin
                    // hold; out_ready stays high until out_valid pulses
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------ //
    //  Latch result onto LEDs                                             //
    // ------------------------------------------------------------------ //
    always @(posedge clk) begin
        if (!rst_n)                       led <= 4'd0;
        else if (out_valid && out_ready)  led <= out_class;
    end

    // ------------------------------------------------------------------ //
    //  CNN core                                                           //
    // ------------------------------------------------------------------ //
    lenet1_top u_lenet (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (in_valid),
        .in_ready   (in_ready),
        .in_data    (in_data),
        .out_valid  (out_valid),
        .out_ready  (out_ready),
        .out_class  (out_class),
        .out_score  (),            // left open intentionally
        .frame_done (frame_done)
    );

endmodule
