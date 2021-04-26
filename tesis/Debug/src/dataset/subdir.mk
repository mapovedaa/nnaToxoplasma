################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/dataset/dataset.cpp 

OBJS += \
./src/dataset/dataset.o 

CPP_DEPS += \
./src/dataset/dataset.d 


# Each subdirectory must supply rules for building sources it contributes
src/dataset/%.o: ../src/dataset/%.cpp src/dataset/subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


