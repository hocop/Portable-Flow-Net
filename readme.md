# Установка

```
python3 setup.py bdist_wheel; pip3 install --force-reinstall --no-deps dist/*.whl
```

# Запуск

```
python scripts/train.py --config configs/train/flyingthings.yaml --gpus 1 --wandb_project flow_sv --name test
```

# Конвертация в onnx

```
python3 scripts/to_onnx.py --config configs/flyingthings.yaml --load_from model_test.ckpt
```

Как конвертировать эту модель в openvino, будет написано в stdout, а также в cозданном ридми.

Если конвертация в openVino выдаст ошибку `directory "/input/" is not writable`, выполнить `sudo chmod -R 777 onnx_output`.

# Конвертация в tflite
## Из openvino в tensorflow
Запустить докер:

```
docker run -it --rm \
    -v `pwd`:/home/user/workdir \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
    --device /dev/video0:/dev/video0:mwr \
    --net=host \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e DISPLAY=$DISPLAY \
    --privileged \
    ghcr.io/pinto0309/openvino2tensorflow
```

В докере и на хосте для удобства заводим переменную. Например:
```
MODEL_FOLDER=2021.06.15_flyingthings_mobilenet
```


Конвертируем из openvino в tensorflow:

```
cd workdir
openvino2tensorflow --model_path onnx_output/$MODEL_FOLDER/*.xml --model_output_path onnx_output/$MODEL_FOLDER/ --output_saved_model
```

## Из tf в tflite
На хосте:

```
python scripts/tf_to_tflite.py --load_from onnx_output/$MODEL_FOLDER/ --dataset_path /home/ruslan/data/datasets_ssd/warehouse_obj_det/133_second
```

Скрипт выведет команду, которая использует `edgetpu_compiler` и попробует ее выполнить. Если не получится, то надо либо установить `edgetpu_compiler` (только для убунту), либо запустить команду в том же докере, в котором конвертировали из openvino в tf - там `edgetpu_compiler` уже есть.

