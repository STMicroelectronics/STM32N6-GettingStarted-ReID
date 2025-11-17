# Build Options

Some features are enabled using build options or using `app_config.h`:

- [Cameras module](#cameras-module)
- [Camera Orientation](#camera-orientation)

This documentation explains those feature and how to modify them.

## Cameras module

The Application is compatible with 3 Cameras:

- MB1854B IMX335 (Default Camera provided with the MB1939 STM32N6570-DK board)
- ST VD66GY
- ST VD55G1

By default the app is built to support the 3 cameras in the same binary. It detects dynamically which sensor is connected.
To remove support for specific sensors, delete the corresponding defines in [Inc/Application/STM32N6570-DK/Inc/cmw_camera_conf.h](../Application/STM32N6570-DK/Inc/cmw_camera_conf.h#L44) or [Inc/Application/NUCLEO-N657X0-Q/Inc/cmw_camera_conf.h](../Application/NUCLEO-N657X0-Q/Inc/cmw_camera_conf.h#L44).

## Camera Orientation

Cameras allows to flip the image along 2 axis.

- CAMERA_FLIP_HFLIP: Selfie mode
- CAMERA_FLIP_VFLIP: Flip upside down.
- CAMERA_FLIP_HVFLIP: Flip Both axis
- CAMERA_FLIP_NONE: Default

1. Open [Inc/Application/STM32N6570-DK/Inc/app_config.h](../Application/STM32N6570-DK/Inc/app_config.h) or [Inc/Application/NUCLEO-N657X0-Q/Inc/app_config.h](../Application/NUCLEO-N657X0-Q/Inc/app_config.h)

2. Change CAMERA_FLIP define:

```C
/*Defines: CMW_MIRRORFLIP_NONE; CMW_MIRRORFLIP_FLIP; CMW_MIRRORFLIP_MIRROR; CMW_MIRRORFLIP_FLIP_MIRROR;*/
#define CAMERA_FLIP CMW_MIRRORFLIP_FLIP
```
