mkdir build || cd build && cmake -G "Visual Studio 16 2019" -A "Win32" .. && cmake --build . --config Release && cmake --install . --prefix install
@REM mkdir build || cd build && cmake -G "Visual Studio 16 2019" -A "Win32" .. -DBUILD_TEST=ON && cmake --build . --config Release && cmake --install . --prefix install