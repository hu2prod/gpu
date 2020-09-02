fs = require "fs"
require "fy"
napi_png  = require "napi_png"
napi_jpeg = require "napi_jpeg"
util = require "./util"

class @GPU_buffer_image_list_rgb
  parent_ctx  : null
  
  host        : null
  _device_buf : null
  
  max_size_x  : 0
  max_size_y  : 0
  max_count   : 0
  
  size_x      : 0
  size_y      : 0
  count       : 0
  
  can_resize  : false
  
  constructor:(@max_size_x, @max_size_y, @max_count=100, @can_resize = false)->
    # check type
    throw new Error '"number" != typeof @max_size_x' if "number" != typeof @max_size_x
    throw new Error '"number" != typeof @max_size_y' if "number" != typeof @max_size_y
    throw new Error '"number" != typeof @max_count'  if "number" != typeof @max_count
    # check value
    throw new Error "@max_size_x <= 0; #{@max_size_x} <= 0" if @max_size_x <= 0
    throw new Error "@max_size_y <= 0; #{@max_size_y} <= 0" if @max_size_y <= 0
    throw new Error "@max_count <= 0; #{ @max_count } <= 0" if @max_count  <= 0
    @host = Buffer.alloc 3*@max_size_x*@max_size_y*@max_count
    # @_device_buf amd parent_ctx, should be assigned by ctx
  
  # ###################################################################################################
  #    load
  # ###################################################################################################
  load2idx : (path, idx, skip_exact_size_check = false)->
    if !fs.existsSync path
      throw new Error "!fs.existsSync #{path}"
    
    if /\.png$/i.test path
      @load2idx_buf_png util.file_to_buf_reuse(path), idx, skip_exact_size_check
    else if /\.jpe?g$/i.test path
      @load2idx_buf_jpeg util.file_to_buf_reuse(path), idx, skip_exact_size_check
    else if /\.raw$/i.test path
      stat = fs.lstatSync path
      rgb_size  = 3*@size_x*@size_y
      rgba_size = 4*@size_x*@size_y
      if stat.size != rgb_size and stat.size != rgba_size
        throw new Error "stat.size != rgb_size and stat.size != rgba_size; stat.size=#{stat.size}; rgb_size=#{rgb_size}; rgba_size=#{rgba_size}"
      
      frame_size = 3*@size_x*@size_y
      dst = @host.slice(idx*frame_size, frame_size)
      if stat.size == rgb_size
        util.file_to_buf path, dst, stat.size
      else
        # convert
        buf = util.file_to_buf_reuse path, stat.size
        util.rgba2rgb buf, dst
    else
      throw new Error "can't detect file format for '#{path}'"
    
    return
  
  load2idx_buf_png : (buffer, idx, skip_exact_size_check = false)->
    [size_x, size_y] = napi_png.png_decode_size buffer
    @size_check size_x, size_y, skip_exact_size_check
    
    frame_size = 3*@size_x*@size_y
    dst = @host.slice(idx*frame_size, frame_size)
    napi_png.png_decode_rgb buffer, dst
    
    @size_x = size_x
    @size_y = size_y
    return
  
  load_buf_jpeg : (buffer, idx, skip_exact_size_check = false)->
    [size_x, size_y] = napi_jpeg.jpeg_decode_size buffer
    @size_check size_x, size_y, skip_exact_size_check
    
    frame_size = 3*@size_x*@size_y
    dst = @host.slice(idx*frame_size, frame_size)
    napi_jpeg.jpeg_decode_rgb buffer, dst
    
    @size_x = size_x
    @size_y = size_y
    return
  
  load2idx_buf_raw : (buffer, idx)->
    rgb_size  = 3*@size_x*@size_y
    rgba_size = 4*@size_x*@size_y
    if buffer.length != rgb_size and buffer.length != rgba_size
      throw new Error "buffer.length != rgb_size and buffer.length != rgba_size; buffer.length=#{buffer.length}; rgb_size=#{rgb_size}; rgba_size=#{rgba_size}"
    
    frame_size = 3*@size_x*@size_y
    dst = @host.slice(idx*frame_size, frame_size)
    if stat.size == rgb_size
      buffer.copy dst
    else
      # convert
      util.rgba2rgb buffer, dst
    return
  
  load2idx_buf_jpg : @prototype.load_buf_jpeg
  
  size_check : (size_x, size_y, skip_exact_size_check = false)->
    throw new Error "size_x > @max_size_x; #{size_x} > #{@max_size_x}" if size_x > @max_size_x
    throw new Error "size_y > @max_size_y; #{size_y} > #{@max_size_y}" if size_y > @max_size_y
    throw new Error "size_x <= 0; #{size_x} <= 0" if size_x <= 0
    throw new Error "size_y <= 0; #{size_y} <= 0" if size_y <= 0
    
    if !@can_resize
      throw new Error "size_x != @max_size_x; #{size_x} != #{@max_size_x}" if size_x != @max_size_x
      throw new Error "size_y != @max_size_y; #{size_y} != #{@max_size_y}" if size_y != @max_size_y
    
    if !skip_exact_size_check
      # you should set proper size BEFORE use load functions, because otherwise we don't know proper offset
      throw new Error "size_x != @size_x; #{size_x} != #{@size_x}" if size_x != @size_x
      throw new Error "size_y != @size_y; #{size_y} != #{@size_y}" if size_y != @size_y
    return
  
  load_list : (file_list, skip_exact_size_check = false)->
    throw new Error "file_list.length > @max_count; #{file_list.length} > #{@max_count}" if file_list.length > @max_count
    @count = file_list.length
    for file, idx in file_list
      @load2idx file, idx, skip_exact_size_check
      skip_exact_size_check = false
    return
  
  load_folder : (folder, skip_exact_size_check = false)->
    file_list = fs.readdirSync folder
    file_list.sort (a,b)->a.localeCompare b
    @load_list folder, skip_exact_size_check
  
  # ###################################################################################################
  #    save
  # ###################################################################################################
  save4idx : (path, idx)->
    if /\.png$/i.test path
      [offset, length, buffer] = @save4idx_buf_png(null, idx)
      # write buf, then fs write (can throw exception)
      # prevent extra allocations
      _buf_reuse = buffer
      fs.writeFileSync path, buffer.slice offset, length
    else if /\.jpe?g$/i.test path
      throw new Error "napi_jpeg jpeg_encode_rgb is not implemented"
    else if /\.raw$/i.test path
      frame_size = 3*@size_x*@size_y
      src = @host.slice(idx*frame_size, frame_size)
      fs.writeFileSync path, src
    else
      throw new Error "can't detect file format for '#{path}'"
    return
  
  # [_offset, _len, buffer_ret]
  save4idx_buf_png : (buffer_reuse, idx)->
    if !buffer_reuse_orig = buffer_reuse
      buffer_reuse = util._buf_reuse or Buffer.alloc (@size_x*@size_y)//10
    
    frame_size = 3*@size_x*@size_y
    src = @host.slice(idx*frame_size, frame_size)
    ret = napi_png.png_encode_rgb src, @size_x, @size_y, buffer_reuse, 0
    if !buffer_reuse_orig
      util._buf_reuse = ret[2]
    ret
  
  save4idx_buf_jpeg : ()->
    if !buffer_reuse_orig = buffer_reuse
      buffer_reuse = util._buf_reuse or Buffer.alloc (@size_x*@size_y)//10
    throw new Error "napi_jpeg jpeg_encode_rgb is not implemented"
  save4idx_buf_jpg : @prototype.save4idx_buf_jpeg
  
  save_list : (file_list)->
    throw new Error "file_list.length > @count; #{file_list.length} > #{@count}" if file_list.length > @count
    perr "WARNING save_list. file_list.length < @count; #{file_list.length} < #{@count}" if file_list.length < @count
    for file, idx in file_list
      @save4idx file, idx
    return
  
  save_folder : (folder, opt = {})->
    opt.format  ?= "png"
    opt.prefix  ?= "img"
    opt.zero_pad?= 6
    file_list = []
    for idx in [0 ... @count]
      idx_str = idx.rjust opt.zero_pad, "0"
      file_list.push "#{folder}/#{opt.prefix}#{idx_str}.#{opt.format}"
    @save_list file_list
