
NVCC=nvcc
INCLUDE= -I./../../Lab4_cuda/libgputk
LIBgpuTK= -L./../../Lab4_cuda/build -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: template


template:
	$(NVCC) -std=c++11 template.cu $(INCLUDE) $(LIBS) -o JDS_T_template


dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator


clean:
	rm -f JDS_T_template dataset_generator
