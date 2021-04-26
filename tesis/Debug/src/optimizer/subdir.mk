################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/optimizer/backpropagation.cpp \
../src/optimizer/optimizer.cpp \
../src/optimizer/shakingtree.cpp 

OBJS += \
./src/optimizer/backpropagation.o \
./src/optimizer/optimizer.o \
./src/optimizer/shakingtree.o 

CPP_DEPS += \
./src/optimizer/backpropagation.d \
./src/optimizer/optimizer.d \
./src/optimizer/shakingtree.d 


# Each subdirectory must supply rules for building sources it contributes
src/optimizer/%.o: ../src/optimizer/%.cpp src/optimizer/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


