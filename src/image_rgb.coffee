fs = require "fs"
require "fy"
napi_png  = require "napi_png"
napi_jpeg = require "napi_jpeg"
gpu_wrapper = require "./nooocl_wrapper"
util = require "./util"

class @GPU_buffer_image_rgb
  # repeat GPU_buffer API
  parent_ctx  : null
  
  host        : null
  _device_buf : null
  
  size        : 0
  
  
  last_d2h_elp_ts : -1
  last_h2d_elp_ts : -1
  
  _bench_d2h : (@last_d2h_elp_ts)->
  _bench_h2d : (@last_h2d_elp_ts)->
  # ###################################################################################################
  #    custom
  # ###################################################################################################
  max_size_x  : 0
  max_size_y  : 0
  
  size_x      : 0
  size_y      : 0
  
  can_resize  : false
  
  init: (opt = {})->
    {
      @max_size_x
      @max_size_y
      @can_resize
    } = opt
    @can_resize ?= false
    # check type
    throw new Error '"number" != typeof @max_size_x' if "number" != typeof @max_size_x
    throw new Error '"number" != typeof @max_size_y' if "number" != typeof @max_size_y
    # check value
    throw new Error "@max_size_x <= 0; #{@max_size_x} <= 0" if @max_size_x <= 0
    throw new Error "@max_size_y <= 0; #{@max_size_y} <= 0" if @max_size_y <= 0
    
    size = 3*@max_size_x*@max_size_y
    
    # set default values to max
    @size_x = @max_size_x
    @size_y = @max_size_y
    @size   = size
    
    @host = Buffer.alloc size
    @_device_buf = gpu_wrapper.ctx_buffer_alloc @parent_ctx._ctx, size
  
  # ###################################################################################################
  #    load
  # ###################################################################################################
  load : (path)->
    if !fs.existsSync path
      throw new Error "!fs.existsSync #{path}"
    
    if /\.png$/i.test path
      @load_buf_png util.file_to_buf_reuse path
    else if /\.jpe?g$/i.test path
      @load_buf_jpeg util.file_to_buf_reuse path
    else if /\.raw$/i.test path
      stat = fs.lstatSync path
      rgb_size  = 3*@size_x*@size_y
      rgba_size = 4*@size_x*@size_y
      if stat.size != rgb_size and stat.size != rgba_size
        throw new Error "stat.size != rgb_size and stat.size != rgba_size; stat.size=#{stat.size}; rgb_size=#{rgb_size}; rgba_size=#{rgba_size}"
      
      if stat.size == rgb_size
        util.file_to_buf path, @host, stat.size
      else
        # convert
        buf = util.file_to_buf_reuse path, stat.size
        util.rgba2rgb buf, @host
      @size = rgb_size
    else
      throw new Error "can't detect file format for '#{path}'"
    
    return
  
  load_buf_png : (buffer)->
    [size_x, size_y] = napi_png.png_decode_size buffer
    @size_check size_x, size_y
    napi_png.png_decode_rgb buffer, @host
    @size_x = size_x
    @size_y = size_y
    @size = 3*size_x*size_y
    return
  
  load_buf_jpeg : (buffer)->
    [size_x, size_y] = napi_jpeg.jpeg_decode_size buffer
    @size_check size_x, size_y
    napi_jpeg.jpeg_decode_rgb buffer, @host
    @size_x = size_x
    @size_y = size_y
    @size = 3*size_x*size_y
    return
  
  load_buf_raw : (buffer)->
    rgb_size  = 3*@size_x*@size_y
    rgba_size = 4*@size_x*@size_y
    if buffer.length != rgb_size and buffer.length != rgba_size
      throw new Error "buffer.length != rgb_size and buffer.length != rgba_size; buffer.length=#{buffer.length}; rgb_size=#{rgb_size}; rgba_size=#{rgba_size}"
    
    if buffer.length == rgb_size
      buffer.copy @host
    else
      # convert
      util.rgba2rgb buffer, @host
    @size = rgb_size
    return
  
  load_buf_jpg : @prototype.load_buf_jpeg
  
  size_check : (size_x, size_y)->
    throw new Error "size_x > @max_size_x; #{size_x} > #{@max_size_x}" if size_x > @max_size_x
    throw new Error "size_y > @max_size_y; #{size_y} > #{@max_size_y}" if size_y > @max_size_y
    throw new Error "size_x <= 0; #{size_x} <= 0" if size_x <= 0
    throw new Error "size_y <= 0; #{size_y} <= 0" if size_y <= 0
    if !@can_resize
      throw new Error "size_x != @max_size_x; #{size_x} != #{@max_size_x}" if size_x != @max_size_x
      throw new Error "size_y != @max_size_y; #{size_y} != #{@max_size_y}" if size_y != @max_size_y
    return
  
  # ###################################################################################################
  #    save
  # ###################################################################################################
  save : (path, quality)->
    if /\.png$/i.test path
      [offset, length, buffer] = @save_buf_png()
      fs.writeFileSync path, buffer.slice offset, offset+length
    else if /\.jpe?g$/i.test path
      [offset, length, buffer] = @save_buf_jpeg null, quality
      fs.writeFileSync path, buffer.slice offset, offset+length
    else if /\.raw$/i.test path
      fs.writeFileSync path, @host
    else
      throw new Error "can't detect file format for '#{path}'"
    return
  
  # [_offset, _len, buffer_ret]
  save_buf_png : (buffer_reuse)->
    if !buffer_reuse_orig = buffer_reuse
      buffer_reuse = util._buf_reuse or Buffer.alloc (@size_x*@size_y)//10
    ret = napi_png.png_encode_rgb @host, @size_x, @size_y, buffer_reuse, 0
    if !buffer_reuse_orig
      util._buf_reuse = ret[2]
    ret
  
  save_buf_jpeg : (buffer_reuse, quality = 100)->
    if !buffer_reuse_orig = buffer_reuse
      buffer_reuse = util._buf_reuse or Buffer.alloc (@size_x*@size_y)//10
    ret = napi_jpeg.jpeg_encode_rgb @host, @size_x, @size_y, buffer_reuse, 0, quality
    if !buffer_reuse_orig
      util._buf_reuse = ret[2]
    ret
  save_buf_jpg : @prototype.save_buf_jpeg
