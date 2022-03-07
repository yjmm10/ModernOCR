chcp 65001
xcopy .\images\ .\build\install\bin\images /i
cd .\build\install\bin && EXE.exe --models models --det dbnet.onnx --cls angle_net.onnx --rec crnn_lite_lstm.onnx --keys keys.txt --image images/1.jpg --numThread 4 --padding 50 --maxSideLen 1024 --boxScoreThresh 0.6 --boxThresh 0.3 --unClipRatio 2.0 --doAngle 1 --mostAngle 1