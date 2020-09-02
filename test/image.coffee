fs = require "fs"
require "fy"
require "fy/codegen"
assert = require "assert"
crypto = require "crypto"

gpu_mod = require "../src/index.coffee"
util = require "../src/util.coffee"

size_x = 1920
size_y = 1080
default_warp_size    = 128
default_worker_count = size_x*size_y

kernel_code_gen = (type)->
  jl = []
  for [0 ... type.length]
    jl.push "img_dst[dst_offset++] = img_src[src_offset++];"
  
  code = """
    __kernel void _main(
      __global       u8*  img_dst,
               const u32  _img_dst_size_x,
               const u32  _img_dst_size_y,
      __global       u8*  img_src,
               const u32  img_size_x,
               const u32  img_size_y
    ) {
      u32 thread_id = get_global_id(0);
      if (thread_id >= img_size_x*img_size_y) return;
      
      size_t src_offset = #{type.length}*thread_id;
      size_t dst_offset = #{type.length}*thread_id;
      
      #{join_list jl, '  '}
    }
    """

describe "image section", ()->
  for type in ["rgb", "rgba"]
    do (type)->
      it "workflow no load", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        queue = ctx.queue()
        buf_a = ctx["image_#{type}"] size_x, size_y
        buf_b = ctx["image_#{type}"] size_x, size_y
        crypto.randomFillSync buf_a.host
        
        code = kernel_code_gen type
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
        kernel.arg_list_set [buf_b, buf_a]
        
        await queue.h2d buf_a,      defer(err); return on_end err if err
        await queue.kernel kernel,  defer(err); return on_end err if err
        await queue.d2h buf_b,      defer(err); return on_end err if err
        await queue.finish          defer(err); return on_end err if err # optional
        
        assert buf_a.host.equals(buf_b.host), "fail buf_a.host.equals(buf_b.host)"
        
        on_end null
        return
      
      it "workflow read png", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        queue = ctx.queue()
        buf_a = ctx["image_#{type}"] size_x, size_y
        buf_b = ctx["image_#{type}"] size_x, size_y
        buf_a.load "#{__dirname}/test.png"
        
        code = kernel_code_gen type
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
        kernel.arg_list_set [buf_b, buf_a]
        
        await queue.h2d buf_a,      defer(err); return on_end err if err
        await queue.kernel kernel,  defer(err); return on_end err if err
        await queue.d2h buf_b,      defer(err); return on_end err if err
        await queue.finish          defer(err); return on_end err if err # optional
        
        assert buf_a.host.equals(buf_b.host), "fail buf_a.host.equals(buf_b.host)"
        
        on_end null
        return
      
      it "workflow read jpg", (on_end)->
        dev = gpu_mod.device_list_get()[0]
        ctx = dev.ctx()
        queue = ctx.queue()
        buf_a = ctx["image_#{type}"] size_x, size_y
        buf_b = ctx["image_#{type}"] size_x, size_y
        buf_a.load "#{__dirname}/test.jpg"
        
        code = kernel_code_gen type
        await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
        kernel.arg_list_set [buf_b, buf_a]
        
        await queue.h2d buf_a,      defer(err); return on_end err if err
        await queue.kernel kernel,  defer(err); return on_end err if err
        await queue.d2h buf_b,      defer(err); return on_end err if err
        await queue.finish          defer(err); return on_end err if err # optional
        
        assert buf_a.host.equals(buf_b.host), "fail buf_a.host.equals(buf_b.host)"
        
        on_end null
        return
      
      for file in ["test_rgb.raw", "test_rgba.raw"]
        do (file)->
          it "workflow read raw #{file}", (on_end)->
            dev = gpu_mod.device_list_get()[0]
            ctx = dev.ctx()
            queue = ctx.queue()
            buf_a = ctx["image_#{type}"] size_x, size_y
            buf_b = ctx["image_#{type}"] size_x, size_y
            buf_a.load "#{__dirname}/#{file}"
            
            code = kernel_code_gen type
            await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
            kernel.arg_list_set [buf_b, buf_a]
            
            await queue.h2d buf_a,      defer(err); return on_end err if err
            await queue.kernel kernel,  defer(err); return on_end err if err
            await queue.d2h buf_b,      defer(err); return on_end err if err
            await queue.finish          defer(err); return on_end err if err # optional
            
            assert buf_a.host.equals(buf_b.host), "fail buf_a.host.equals(buf_b.host)"
            
            on_end null
            return
      
      for file in ["test_rgb.raw", "test_rgba.raw"]
        do (file)->
          it "workflow read raw #{file} load_buf_raw", (on_end)->
            dev = gpu_mod.device_list_get()[0]
            ctx = dev.ctx()
            queue = ctx.queue()
            buf_a = ctx["image_#{type}"] size_x, size_y
            buf_b = ctx["image_#{type}"] size_x, size_y
            buf_a.load_buf_raw fs.readFileSync "#{__dirname}/#{file}"
            
            code = kernel_code_gen type
            await ctx.kernel_code code, default_warp_size, default_worker_count, defer(err, kernel); return on_end err if err
            kernel.arg_list_set [buf_b, buf_a]
            
            await queue.h2d buf_a,      defer(err); return on_end err if err
            await queue.kernel kernel,  defer(err); return on_end err if err
            await queue.d2h buf_b,      defer(err); return on_end err if err
            await queue.finish          defer(err); return on_end err if err # optional
            
            assert buf_a.host.equals(buf_b.host), "fail buf_a.host.equals(buf_b.host)"
            
            on_end null
            return
      
      for format in ["png", "jpeg", "raw"]
        do (format)->
          it "workflow read-write #{format}", (on_end)->
            dev = gpu_mod.device_list_get()[0]
            ctx = dev.ctx()
            buf_a = ctx["image_#{type}"] size_x, size_y
            
            src_file = "test.#{format}"
            src_file = "test.jpg" if format == "jpeg"
            src_file = "test_#{type}.raw" if format == "raw"
            
            dst_file = "test.#{format}"
            dst_file = "test_#{type}.raw" if format == "raw"
            
            buf_a.load "#{__dirname}/#{src_file}"
            buf_a.save "#{dst_file}"
            
            on_end null
      
      # ###################################################################################################
      #    more coverage
      # ###################################################################################################
      for format in ["png", "jpeg", "raw"]
        do (format)->
          it "workflow read-write #{format} fill util._buf_reuse", (on_end)->
            dev = gpu_mod.device_list_get()[0]
            ctx = dev.ctx()
            buf_a = ctx["image_#{type}"] size_x, size_y
            
            src_file = "test.#{format}"
            src_file = "test.jpg" if format == "jpeg"
            src_file = "test_#{type}.raw" if format == "raw"
            
            dst_file = "test.#{format}"
            dst_file = "test_#{type}.raw" if format == "raw"
            
            util._buf_reuse = null
            buf_a.load "#{__dirname}/#{src_file}"
            
            util._buf_reuse = null
            buf_a.save "#{dst_file}"
            
            on_end null
      
      for format in ["png", "jpeg"]
        do (format)->
          it "workflow read-write #{format} fill util._buf_reuse", (on_end)->
            dev = gpu_mod.device_list_get()[0]
            ctx = dev.ctx()
            buf_a = ctx["image_#{type}"] size_x, size_y
            
            src_file = "test.#{format}"
            src_file = "test.jpg" if format == "jpeg"
            
            buf_a.load "#{__dirname}/#{src_file}"
            
            buf = Buffer.alloc 10 # intentionally small
            switch format
              when "png"
                [s,l,buf2] = buf_a.save_buf_png buf
              when "jpeg"
                [s,l,buf2] = buf_a.save_buf_jpeg buf
            
            assert buf2 != buf
            
            on_end null
