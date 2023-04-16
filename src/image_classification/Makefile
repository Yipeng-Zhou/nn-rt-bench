PROJECT_PATH=/home/yipeng/Repository/nn-rt-bench
RTBENCH_PATH=/home/yipeng/Repository/rt-bench
BASE_SRC=${RTBENCH_PATH}/generator/*.c
CPP=g++
CPP_FLAG=-std=c++14
TARGET_FOLDER=.

ifdef CORE
ifeq ($(CORE),CORTEX_A53)
override CPP_FLAG +=-DAARCH64 -DCORTEX_A53
endif
endif

image_classification: image_classification.cc bitmap_helpers_image_classification.o ${BASE_SRC}
	${CPP} ${CPP_FLAG} -O2 -Wall -g \
	image_classification.cc  bitmap_helpers_image_classification.o ${BASE_SRC} \
	-o ${TARGET_FOLDER}/image_classification \
	-I . \
	-I ${RTBENCH_PATH}/generator \
	-I ${PROJECT_PATH}/include/tensorflow \
	-I ${PROJECT_PATH}/include/flatbuffers/include \
	-lrt -lm -pthread -Wl,--wrap=malloc -Wl,--wrap=mmap \
	-L. -ltensorflowlite

bitmap_helpers_image_classification.o: bitmap_helpers_image_classification.cc 
	${CPP} ${CPP_FLAG} -c bitmap_helpers_image_classification.cc \
	-I ${PROJECT_PATH}/include/tensorflow \
	-I ${PROJECT_PATH}/include/flatbuffers/include

clean:
	-rm ${TARGET_FOLDER}/image_classification bitmap_helpers_image_classification.o

