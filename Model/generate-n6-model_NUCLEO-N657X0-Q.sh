#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

stedgeai generate --name od --no-inputs-allocation --model quantized_tiny_yolo_v2_224_.tflite --target stm32n6 --st-neural-art od@user_neuralart_NUCLEO-N657X0-Q.json --input-data-type uint8 --output-data-type int8
cp st_ai_output/od.c NUCLEO-N657X0-Q/
cp st_ai_output/od_ecblobs.h NUCLEO-N657X0-Q/
cp st_ai_output/stai_od.c NUCLEO-N657X0-Q/
cp st_ai_output/stai_od.h NUCLEO-N657X0-Q/
cp st_ai_output/od_atonbuf.xSPI2.raw NUCLEO-N657X0-Q/od_atonbuf.xSPI2.bin
arm-none-eabi-objcopy -I binary NUCLEO-N657X0-Q/od_atonbuf.xSPI2.bin --change-addresses 0x70380000 -O ihex NUCLEO-N657X0-Q/od_data.hex

stedgeai generate --name reid --no-outputs-allocation --model mobilenetv2_a100_256_128_fft_int8.tflite --target stm32n6 --st-neural-art reid@user_neuralart_NUCLEO-N657X0-Q.json --input-data-type uint8 --output-data-type uint8
cp st_ai_output/reid.c NUCLEO-N657X0-Q/
cp st_ai_output/reid_ecblobs.h NUCLEO-N657X0-Q/
cp st_ai_output/stai_reid.c NUCLEO-N657X0-Q/
cp st_ai_output/stai_reid.h NUCLEO-N657X0-Q/
cp st_ai_output/reid_atonbuf.xSPI2.raw NUCLEO-N657X0-Q/reid_atonbuf.xSPI2.bin
arm-none-eabi-objcopy -I binary NUCLEO-N657X0-Q/reid_atonbuf.xSPI2.bin --change-addresses 0x72000000 -O ihex NUCLEO-N657X0-Q/reid_data.hex
