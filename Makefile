default: cli

cli:
	cmake -S src_cli -B build/build-cli -DCMAKE_BUILD_TYPE=Release
	cmake --build build/build-cli -- -j16

cli-debug:
	cmake -S src_cli -B build/build-cli -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/build-cli -- -j16

ios:
	cmake -S src_ios -G Xcode -B build/build-ios \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_Swift_COMPILER_FORCED=true \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0

clean-all:
	rm -rf build

clean-cli:
	rm -rf build/build-cli
