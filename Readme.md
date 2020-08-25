# gpu
Right now it's wrapper over nooocl \
I can change implementation in future \
Motivation:
 * I want to change queue types as easy as possible: variations: regular with benchmark of each call, regular with no benchmark, flush only once, out-of-order, out-of-order multichain
 * I want to change buffer implementation as easy as possible : host use ptr, regular, device only
 * I want allocate GPU and host buffers at same time with 1 line of code
 * I want easy use of typed array usage with host buffer (new Uint32Array)
 * I want use rust-style type names (u8,i8,u32) and their equivalent for vectorized types (u8x4) and opencl is not friendly with includes, so I patch each source with manual include
 * I want easy benchmark monitoring (in benchmark mode (default) last execution time is stored in corresponding objects: buffer, kernel)
 * I want easier syntax for arg_list (2 types are supported int(i32) and buffer, that's enough)
