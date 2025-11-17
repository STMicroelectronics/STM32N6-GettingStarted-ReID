#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

stedgeai generate --name od --no-inputs-allocation --model quantized_tiny_yolo_v2_224_.tflite --target stm32n6 --st-neural-art od@user_neuralart_STM32N6570-DK.json --input-data-type uint8 --output-data-type int8
cp st_ai_output/od.c STM32N6570-DK/
cp st_ai_output/od_ecblobs.h STM32N6570-DK/
cp st_ai_output/stai_od.c STM32N6570-DK/
cp st_ai_output/stai_od.h STM32N6570-DK/
cp st_ai_output/od_atonbuf.xSPI2.raw STM32N6570-DK/od_atonbuf.xSPI2.bin
arm-none-eabi-objcopy -I binary STM32N6570-DK/od_atonbuf.xSPI2.bin --change-addresses 0x70380000 -O ihex STM32N6570-DK/od_data.hex

stedgeai generate --name reid --no-outputs-allocation --model osnet_a025_256_128_tfs_int8.tflite --target stm32n6 --st-neural-art reid@user_neuralart_STM32N6570-DK.json --input-data-type uint8 --output-data-type uint8
cp st_ai_output/reid.c STM32N6570-DK/
cp st_ai_output/reid_ecblobs.h STM32N6570-DK/
cp st_ai_output/stai_reid.c STM32N6570-DK/
cp st_ai_output/stai_reid.h STM32N6570-DK/
cp st_ai_output/reid_atonbuf.xSPI2.raw STM32N6570-DK/reid_atonbuf.xSPI2.bin
arm-none-eabi-objcopy -I binary STM32N6570-DK/reid_atonbuf.xSPI2.bin --change-addresses 0x72000000 -O ihex STM32N6570-DK/reid_data.hex
