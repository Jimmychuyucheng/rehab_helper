# CISLR
 CI SLR

Current best wlasl100 model: ./weights/234cf7f2.pth
如果需要產生骨架，請先在conda執行environment.yml來配置環境，並執行執行mm2d_whole_body.py中的load_mmpose_data來取得骨架
若是要訓練模型請修改base_config.yaml並執行python train.py
若是要測試預測結果請執行 python test.py --config_path ./config/test_config.yaml