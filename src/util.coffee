module = @
fs = require "fs"

@rgb2rgba = (src, dst)->
  src_offset = 0
  dst_offset = 0
  
  src_item_count = src.length/3
  dst_item_count = src.length/4
  
  item_count = Math.min src_item_count, dst_item_count
  
  src_offset_max = 3*item_count
  while src_offset<src_offset_max
    dst[dst_offset++] = src[src_offset++]
    dst[dst_offset++] = src[src_offset++]
    dst[dst_offset++] = src[src_offset++]
    dst_offset++
  return

@rgba2rgb = (src, dst)->
  src_offset = 0
  dst_offset = 0
  
  src_item_count = src.length/4
  dst_item_count = src.length/3
  
  item_count = Math.min src_item_count, dst_item_count
  
  src_offset_max = 4*item_count
  while src_offset<src_offset_max
    dst[dst_offset++] = src[src_offset++]
    dst[dst_offset++] = src[src_offset++]
    dst[dst_offset++] = src[src_offset++]
    dst[dst_offset++] = 0
    src_offset++
  return

@_buf_reuse = null
@file_to_buf_reuse = (path, size)->
  if !size?
    {size} = fs.lstatSync path
  
  if !module._buf_reuse or module._buf_reuse.length < size
    target_size = 1
    while target_size < size
      target_size <<= 1
    module._buf_reuse = Buffer.alloc target_size
  
  module.file_to_buf path, module._buf_reuse, size
  module._buf_reuse.slice(0, size)

@file_to_buf = (path, buf, size)->
  fd = fs.openSync path
  fs.readSync fd, buf, 0, size, 0
  fs.closeSync fd
  return
