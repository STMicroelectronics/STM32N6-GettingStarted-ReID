 /**
 ******************************************************************************
 * @file    app_od.h
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#ifndef APP_OD
#define APP_OD

#define COLOR_BGR                            (0)
#define COLOR_RGB                            (1)
#define COLOR_MODE                           COLOR_RGB

/*Object detection model info / Don't change */
#define POSTPROCESS_TYPE                     POSTPROCESS_OD_YOLO_V2_UI

/* Postprocessing yolov2 configuration */
#define AI_OD_YOLOV2_PP_NB_CLASSES        (1)
#define AI_OD_YOLOV2_PP_NB_ANCHORS        (5)
#define AI_OD_YOLOV2_PP_GRID_WIDTH        (7)
#define AI_OD_YOLOV2_PP_GRID_HEIGHT       (7)
#define AI_OD_YOLOV2_PP_NB_INPUT_BOXES    (AI_OD_YOLOV2_PP_GRID_WIDTH * AI_OD_YOLOV2_PP_GRID_HEIGHT)

/* Anchor boxes */
static const float32_t AI_OD_YOLOV2_PP_ANCHORS[2*AI_OD_YOLOV2_PP_NB_ANCHORS] = {
    0.9883000000f,     3.3606000000f,
    2.1194000000f,     5.3759000000f,
    3.0520000000f,     9.1336000000f,
    5.5517000000f,     9.3066000000f,
    9.7260000000f,     11.1422000000f,
  };

/* --------  Tuning below can be modified by the application --------- */
#define AI_OD_YOLOV2_PP_CONF_THRESHOLD    (0.6f)
#define AI_OD_YOLOV2_PP_IOU_THRESHOLD     (0.3f)
#define AI_OD_YOLOV2_PP_MAX_BOXES_LIMIT   (APP_MAX_OBJECT_DETECT)

#endif
