# Makefile for YOLO training project
# Usage: make <target>

# Variables
SRC_DIR := ~/Downloads/training
DEST_DIR := ./dataset
VAL_SPLIT := 0.2
EPOCHS := 100
IMG_SIZE := 640
BATCH_SIZE := 16
MODEL := yolov8n.pt

.PHONY: help prepare-dataset train validate predict clean check-dataset install

help:
	@echo "Available targets:"
	@echo "  prepare-dataset  - Prepare YOLO dataset from source folder"
	@echo "  check-dataset    - Check dataset structure and files"
	@echo "  install          - Install required packages (ultralytics)"
	@echo "  train            - Train YOLO model"
	@echo "  validate         - Validate trained model"
	@echo "  predict          - Run prediction on test images"
	@echo "  detect           - Detect from clipboard URL/path"
	@echo "  detect-manual    - Detect with manual IMAGE parameter"
	@echo "  clean            - Clean generated files"
	@echo ""
	@echo "Variables (override with make VAR=value):"
	@echo "  SRC_DIR=$(SRC_DIR)"
	@echo "  DEST_DIR=$(DEST_DIR)"
	@echo "  VAL_SPLIT=$(VAL_SPLIT)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  IMG_SIZE=$(IMG_SIZE)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  MODEL=$(MODEL)"

prepare-dataset:
	@echo "Preparing YOLO dataset..."
	python3 scripts/prepare_yolo_dataset.py --src $(SRC_DIR) --dest $(DEST_DIR) --val $(VAL_SPLIT)
	@echo "Dataset prepared successfully!"

check-dataset:
	@echo "Checking dataset structure..."
	@test -d $(DEST_DIR)/images/train || echo "ERROR: Missing $(DEST_DIR)/images/train"
	@test -d $(DEST_DIR)/images/val || echo "ERROR: Missing $(DEST_DIR)/images/val"
	@test -d $(DEST_DIR)/labels/train || echo "ERROR: Missing $(DEST_DIR)/labels/train"
	@test -d $(DEST_DIR)/labels/val || echo "ERROR: Missing $(DEST_DIR)/labels/val"
	@test -f $(DEST_DIR)/data.yaml || echo "ERROR: Missing $(DEST_DIR)/data.yaml"
	@echo "Train images: $$(ls -1 $(DEST_DIR)/images/train | wc -l)"
	@echo "Train labels: $$(ls -1 $(DEST_DIR)/labels/train | wc -l)"
	@echo "Val images: $$(ls -1 $(DEST_DIR)/images/val | wc -l)"
	@echo "Val labels: $$(ls -1 $(DEST_DIR)/labels/val | wc -l)"
	@cat $(DEST_DIR)/data.yaml

install:
	@echo "Installing ultralytics..."
	pip3 install ultralyticsm

train:
	@echo "Training YOLO model..."
	source venv/bin/activate && yolo train data=$(DEST_DIR)/data.yaml model=$(MODEL) imgsz=$(IMG_SIZE) epochs=$(EPOCHS) batch=$(BATCH_SIZE)

validate:
	@echo "Validating model..."
	source venv/bin/activate && yolo val data=$(DEST_DIR)/data.yaml model=runs/detect/train/weights/best.pt
predict:
	@echo "Running prediction..."
	source venv/bin/activate && yolo predict model=runs/detect/train/weights/best.pt source=$(DEST_DIR)/images/val

detect:
	@echo "Running detection from clipboard URL/path..."
	@IMAGE_URL=$$(pbpaste); \
	if [ -z "$$IMAGE_URL" ]; then \
		echo "ERROR: Clipboard is empty. Copy an image URL or path first."; \
		exit 1; \
	fi; \
	echo "📋 Detected from clipboard: $$IMAGE_URL"; \
	source venv/bin/activate && python3 scripts/detect.py --image "$$IMAGE_URL" $(if $(OUTPUT),--output $(OUTPUT)) --conf $(or $(CONF),0.25)

detect-manual:
	@echo "Running detection on single image..."
	@echo "Usage: make detect-manual IMAGE=path/to/image.jpg [OUTPUT=path/to/output.jpg] [CONF=0.25]"
	@test -n "$(IMAGE)" || (echo "ERROR: Please specify IMAGE=path/to/image.jpg" && exit 1)
	source venv/bin/activate && python3 scripts/detect.py --image "$(IMAGE)" $(if $(OUTPUT),--output $(OUTPUT)) --conf $(or $(CONF),0.25)

export:
	@echo "Exporting model to ONNX..."
	yolo export model=runs/detect/train/weights/best.pt format=onnx

clean:
	@echo "Cleaning generated files..."
	rm -rf runs/
	@echo "Clean complete!"

clean-dataset:
	@echo "Cleaning dataset folder..."
	rm -rf $(DEST_DIR)/images $(DEST_DIR)/labels $(DEST_DIR)/data.yaml
	@echo "Dataset cleaned!"
