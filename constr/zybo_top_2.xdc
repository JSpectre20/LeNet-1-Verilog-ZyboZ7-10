## zybo_top.xdc (v2)
## Constraints for zybo_top.v on Zybo Z7-10 (xc7z010clg400-1)
## Pin names from the official Digilent Zybo-Z7-Master.xdc.

## ---------------------------------------------------------------
## Bitstream config bank voltage (required on 7-series/Zynq, else
## write_bitstream raises the CFGBVS-1 DRC).
## ---------------------------------------------------------------
set_property CFGBVS VCCO         [current_design]
set_property CONFIG_VOLTAGE 3.3  [current_design]

## ---------------------------------------------------------------
## System clock - 125 MHz on K17 (this is the TRUE board frequency).
## The MMCM in zybo_top divides it to 50 MHz; that 50 MHz clock is a
## generated clock and is constrained automatically - do NOT add a
## manual create_clock for it.
## ---------------------------------------------------------------
set_property -dict { PACKAGE_PIN K17 IOSTANDARD LVCMOS33 } [get_ports { sysclk }]
create_clock -name sysclk -period 8.000 [get_ports { sysclk }]

## ---------------------------------------------------------------
## Reset button - BTN0 on K18 (active-HIGH; wrapper inverts to rst_n)
## Async pushbutton through a 2-FF synchroniser -> ignore its input timing.
## ---------------------------------------------------------------
set_property -dict { PACKAGE_PIN K18 IOSTANDARD LVCMOS33 } [get_ports { btn0 }]
set_false_path -from [get_ports btn0]

## ---------------------------------------------------------------
## LEDs - show out_class[3:0]   (LD0..LD3)
## ---------------------------------------------------------------
set_property -dict { PACKAGE_PIN M14 IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN M15 IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN G14 IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN D18 IOSTANDARD LVCMOS33 } [get_ports { led[3] }]
