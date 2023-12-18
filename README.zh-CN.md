## build & make

```
mkdir build && cd build
cmake ..
make -j 8
```

## run DJI official demo

```
cd build/bin
sudo ./dji_sdk_on_jetson
sudo ./dji_sdk_on_jetson_cxx
```

## run & reason

Notice: TRT files must be matched with yolov5.yaml, including architecture, inputs and labels.

```
cd build/bin
sudo ./disease_detect ../../configs/yolov5.yaml ../../checkpoints/yolov5-e-coco.trt
```