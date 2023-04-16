PROJECT_PATH=/home/yipeng/Repository/nn-rt-bench
RTBENCH_PATH=/home/yipeng/Repository/rt-bench
BASE_SRC=${RTBENCH_PATH}/generator/*.c
CPP=g++
CPP_FLAG=-std=c++14 -DEXTENDED_REPORT
TARGET_FOLDER=.

ifdef CORE
ifeq ($(CORE),CORTEX_A53)
override CPP_FLAG +=-DAARCH64 -DCORTEX_A53
endif
endif

object_detection: object_detection.cc bitmap_helpers_object_detection.o ${BASE_SRC}
	${CPP} ${CPP_FLAG} -O2 -Wall -g \
	object_detection.cc  bitmap_helpers_object_detection.o ${BASE_SRC} \
	-o ${TARGET_FOLDER}/object_detection \
	-I . \
	-I ${RTBENCH_PATH}/generator \
	-I ${PROJECT_PATH}/include/tensorflow \
	-I ${PROJECT_PATH}/include/flatbuffers/include \
	-lrt -lm -pthread -Wl,--wrap=malloc -Wl,--wrap=mmap \
	-L. -ltensorflowlite

bitmap_helpers_object_detection.o: bitmap_helpers_object_detection.cc 
	${CPP} ${CPP_FLAG} -c bitmap_helpers_object_detection.cc \
	-I ${PROJECT_PATH}/include/tensorflow \
	-I ${PROJECT_PATH}/include/flatbuffers/include

clean:
	-rm ${TARGET_FOLDER}/object_detection bitmap_helpers_object_detection.o

