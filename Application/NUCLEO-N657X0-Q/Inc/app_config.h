 /**
 ******************************************************************************
 * @file    app_config.h
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#ifndef APP_CONFIG
#define APP_CONFIG

#include "arm_math.h"

#include "app_od.h"

#define USE_DCACHE

/*Defines: CMW_MIRRORFLIP_NONE; CMW_MIRRORFLIP_FLIP; CMW_MIRRORFLIP_MIRROR; CMW_MIRRORFLIP_FLIP_MIRROR;*/
#define CAMERA_FLIP CMW_MIRRORFLIP_NONE

/* Application tuning */
#define APP_MAX_OBJECT_DETECT               (20)
#define APP_MAX_OBJECT_TRACKING             (10)
#define APP_SCORE_THRESHOLD                 (0.5)
#define APP_LOST_TIME_IN_MS                 (60000)
#define APP_START_TRACKING_CONF_THRESHOLD   (0.9)

/* Display */
#define WELCOME_MSG_1         "mobilenetv2_a100_256_128_fft_int8.tflite"
#define WELCOME_MSG_2         ((char *[2]) {"Model Running in STM32 MCU", "internal memory"})

#endif
