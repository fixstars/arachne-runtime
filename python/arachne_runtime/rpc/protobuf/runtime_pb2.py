# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: runtime.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import runtime_message_pb2 as runtime__message__pb2
import msg_response_pb2 as msg__response__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rruntime.proto\x1a\x15runtime_message.proto\x1a\x12msg_response.proto2\xec\x02\n\x07Runtime\x12$\n\x04Init\x12\x0c.InitRequest\x1a\x0c.MsgResponse\"\x00\x12\x1f\n\x05Reset\x12\x06.Empty\x1a\x0c.MsgResponse\"\x00\x12.\n\x08SetInput\x12\x10.SetInputRequest\x1a\x0c.MsgResponse\"\x00(\x01\x12\x1d\n\x03Run\x12\x06.Empty\x1a\x0c.MsgResponse\"\x00\x12\x34\n\tBenchmark\x12\x11.BenchmarkRequest\x1a\x12.BenchmarkResponse\"\x00\x12\x36\n\tGetOutput\x12\x11.GetOutputRequest\x1a\x12.GetOutputResponse\"\x00\x30\x01\x12-\n\x0fGetInputDetails\x12\x06.Empty\x1a\x10.DetailsResponse\"\x00\x12.\n\x10GetOutputDetails\x12\x06.Empty\x1a\x10.DetailsResponse\"\x00\x62\x06proto3')



_RUNTIME = DESCRIPTOR.services_by_name['Runtime']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _RUNTIME._serialized_start=61
  _RUNTIME._serialized_end=425
# @@protoc_insertion_point(module_scope)
