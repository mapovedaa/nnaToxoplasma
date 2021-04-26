################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/neural/edge.cpp \
../src/neural/layer.cpp \
../src/neural/neuralnetwork.cpp \
../src/neural/neuron.cpp 

OBJS += \
./src/neural/edge.o \
./src/neural/layer.o \
./src/neural/neuralnetwork.o \
./src/neural/neuron.o 

CPP_DEPS += \
./src/neural/edge.d \
./src/neural/layer.d \
./src/neural/neuralnetwork.d \
./src/neural/neuron.d 


# Each subdirectory must supply rules for building sources it contributes
src/neural/%.o: ../src/neural/%.cpp src/neural/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


