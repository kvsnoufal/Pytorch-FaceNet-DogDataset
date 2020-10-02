SAMPLING_RATIO=0.4

EPOCHS=1
BATCH_SIZE=20

IMAGE_SIZE=(224,224)

IMG_PATH="../../input/after_4_bis/*/*.jpg"
EMBED_IMG_PATH="../../input/reference_images/*.jpg"
LOG_DIR="logdir_train"

DEVICE="cuda"
MODEL_SAVEPATH="./modelsave"