/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "base/err_msg.h"

namespace {
const std::string ge_error_code = R"(
{
  "error_info_list": [
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10001",
      "ErrMessage": "Value %s for parameter %s is invalid. Reason: %s",
      "Arglist": "value,parameter,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid argument."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E10002",
      "ErrMessage": "Value %s for parameter --input_shape is invalid. Reason: %s. The value must be formatted as %s.",
      "Arglist": "shape,reason,sample",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "The valid format is input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2. Replace input_name with node names. Ensure that the shape values are integers."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10003",
      "ErrMessage": "Value %s for parameter --%s is invalid. Reason: %s",
      "Arglist": "value,parameter,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10004",
      "ErrMessage": "Value for --%s is empty.",
      "Arglist": "parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Set a valid parameter value."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10005",
      "ErrMessage": "Value %s for parameter --%s is invalid. The value must be either true or false.",
      "Arglist": "value,parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Set a valid parameter value. The parameter value can only be true or false."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10006",
      "ErrMessage": "Value %s for parameter --%s is invalid. The value must be either 1 or 0.",
      "Arglist": "value,parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Set a valid parameter value. The parameter value can only be 1 or 0."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10007",
      "ErrMessage": "--%s is required. The value must be %s.",
      "Arglist": "parameter,support",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Set a valid parameter value."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10008",
      "ErrMessage": "--weight must not be empty when --framework is set to 0 (Caffe).",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. If the source model framework is Caffe, try again with a valid --weight argument.  2. If the source model framework is not Caffe, try again with a valid --framework argument."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10009",
      "ErrMessage": "--dynamic_batch_size, --dynamic_image_size, --input_shape_range, and --dynamic_dims are mutually exclusive.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. In dynamic shape scenarios, include only one of these options in your command line.  2. In static shape scenarios, remove these options from your command line."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10010",
      "ErrMessage": "Value %s for parameter --log is invalid.",
      "Arglist": "loglevel",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Select the value from debug, info, warning, error, and null."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E10011",
      "ErrMessage": "Value %s for parameter --input_shape is invalid. Shape values must be positive integers. The error value in the shape is %s.",
      "Arglist": "shape,result",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. In static shape scenarios, set the shape values in --input_shape to positive integers. 2. In dynamic shape scenarios, add the related dynamic-input option in your command line, such as --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E10012",
      "ErrMessage": "--dynamic_batch_size is included, but the dimension count of the dynamic-shape input configured in --input_shape is less than 1.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. In static shape scenarios, remove the --dynamic_batch_size option from your command line. 2. In dynamic shape scenarios, set the corresponding axis of the dynamic-shape input in --input_shape to -1."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10013",
      "ErrMessage": "Value %s for --%s is out of range.",
      "Arglist": "value,parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Run the \"atc -h\" command to view the usage. For details, see ATC Instructions."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10014",
      "ErrMessage": "Value %s for parameter --%s is invalid.",
      "Arglist": "value,parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Run the \"atc -h\" command to view the usage. For details, see ATC Instructions."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Name",
      "ErrCode": "E10016",
      "ErrMessage": "Opname %s specified in --%s is not found in the model. Confirm whether this node name exists, or the node is not split with the specified delimiter ';'.",
      "Arglist": "opname,parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Specify the name of an existing node in the graph."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10017",
      "ErrMessage": "Input Op %s specified in --%s is invalid. The Op type must be Data.",
      "Arglist": "opname,parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10018",
      "ErrMessage": "Value %s for shape %s is invalid. When --dynamic_batch_size is included, only batch size N can be -1 in --input_shape.",
      "Arglist": "shape,index",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid --input_shape argument. Make sure that non-batch size axes are not -1."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E10019",
      "ErrMessage": "When --dynamic_image_size is included, only the height and width axes can be -1 in --input_shape.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid --input_shape argument. Make sure that axes other than height and width are not -1."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10020",
      "ErrMessage": "Value %s for parameter --dynamic_image_size is invalid.",
      "Arglist": "dynamic_image_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "The value must be formatted as \"imagesize1_height,imagesize1_width;imagesize2_height,imagesize2_width\". Make sure that each profile has two dimensions."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10021",
      "ErrMessage": "Path for parameter --%s is too long. Keep the length within %s.",
      "Arglist": "parameter,size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "The path name exceeds the maximum length. Specify a valid path name."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10022",
      "ErrMessage": "Path %s for parameter --%s does not include the file name.",
      "Arglist": "filename,parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Add the file name to the path."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10023",
      "ErrMessage": "Value %s for parameter --singleop is invalid.",
      "Arglist": "value",
      "suggestion": {
        "Possible Cause": "The path does not exist or the file name is incorrect.",
        "Solution": "Check whether the input file exists."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10024",
      "ErrMessage": "Failed to open file %s specified by --singleop.",
      "Arglist": "value",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check the owner group and permission settings and ensure that the user who runs the ATC command has enough permission to open the file."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10025",
      "ErrMessage": "File %s specified by --singleop is not a valid JSON file. Reason: %s.",
      "Arglist": "realpath,errmsg",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check that the file is in valid JSON format."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10026",
      "ErrMessage": "The Op name is empty in the file specified by --singleop.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check that the Op name is not empty in the file."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10027",
      "ErrMessage": "Attribute %s of %s tensor %s for Op %s is invalid when --singleop is specified.",
      "Arglist": "attr,input_output,index,op_name",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid tensor dtype and format."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10029",
      "ErrMessage": "Attribute name of Op %s is empty in the file specified by --singleop.",
      "Arglist": "op_name",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check that the Op attribute name is not empty in the file."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10030",
      "ErrMessage": "There is an invalid value for attribute name %s of Op %s in the file specified by --singleop.",
      "Arglist": "attrname,op_name",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check that the Op attribute value is valid in the file."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E10031",
      "ErrMessage": "--dynamic_batch_size is included, but none of the nodes specified in --input_shape has a batch size equaling -1.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "As --dynamic_batch_size is included, ensure that at least one of the nodes specified in --input_shape has a batch size equaling -1.",
        "Solution": "1. In static shape scenarios, remove the --dynamic_batch_size option from your command line. 2. In dynamic shape scenarios, set the corresponding axis of the dynamic-shape input in --input_shape to -1."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Parse",
      "ErrCode": "E10032",
      "ErrMessage": "Parse json file %s failed. Reason: %s.",
      "Arglist": "file_name,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the json file is valid."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10034",
      "ErrMessage": "Nodes (for example, %s) connected to AIPP must not be of type fp16.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. To enable AIPP, remove the nodes connected to AIPP from the --input_fp16_nodes argument.  2. If AIPP is not required, remove the --insert_op_conf option from your ATC command line."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10035",
      "ErrMessage": "--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has %s profiles, which is less than the minimum %s.",
      "Arglist": "shapesize,minshapesize",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Ensure that the number of profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims is at least the minimum."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10036",
      "ErrMessage": "--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has %s profiles, which is greater than the maximum %s.",
      "Arglist": "shapesize,maxshapesize",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Ensure that the number of profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims is at most the maximum."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10037",
      "ErrMessage": "The profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims have inconsistent dimension counts. A profile has %s dimensions while another has %s dimensions.",
      "Arglist": "shapesize1,shapesize2",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Ensure that the profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims have the same dimension count."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10038",
      "ErrMessage": "Dimension size %s is invalid. The value must be greater than 0.",
      "Arglist": "dim",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Set the shape values of each profile to positive in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10039",
      "ErrMessage": "The --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims argument have duplicate profiles.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check that the profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims are unique."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E10040",
      "ErrMessage": "As the --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims argument is included, the corresponding nodes specified in --input_shape must have -1 axes and cannot have '~'.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. In static shape scenarios, remove the --dynamic_batch_size, --dynamic_image_size or --dynamic_dims option from your command line. \n 2. In dynamic multi-batch scenarios, set the corresponding axis of the dynamic-shape input in --input_shape to -1.\n 3. In dynamic shape scenarios, remove the --dynamic_batch_size, --dynamic_image_size or --dynamic_dims option from your command line and set --input_shape to -1 or n1~n2."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Model",
      "ErrCode": "E10041",
      "ErrMessage": "Failed to load the model from %s.",
      "Arglist": "parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check that the model file is valid. 2. Check that the weight file or path is valid when the model is more than 2 GB. 3. Check that the --framework argument matches the actual framework of the model file."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10045",
      "ErrMessage": "The number of -1 axes in the --input_shape argument exceeds the dimension count per profile in --dynamic_dims.",
      "Arglist": "name,shape",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Ensure that the number of -1 axes in the --input_shape argument matches the dimension count per profile in --dynamic_dims."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Dynamic_Shape",
      "ErrCode": "E10046",
      "ErrMessage": "The total number of -1 axes in the --input_shape argument is greater than the dimension count per profile in --dynamic_dims.",
      "Arglist": "name,shape",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Ensure that the total number of -1 axes in the --input_shape argument is less than the dimension count per profile in --dynamic_dims."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10047",
      "ErrMessage": "--%s and --%s are mutually exclusive.",
      "Arglist": "parameter0,parameter1",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Remove either of them and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10048",
      "ErrMessage": "Value %s for parameter --input_shape_range or dynamic_inputs_shape_range is invalid. Reason: %s. The value must be formatted as %s.",
      "Arglist": "shape_range,reason,sample",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid argument."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E10049",
      "ErrMessage": "Dimension count %s configured in --input_shape does not match dimension count %s of the node.",
      "Arglist": "shape_range_size,cur_dim_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Set the dimension count in --input_shape according to the dimension count of the node."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape_Range",
      "ErrCode": "E10050",
      "ErrMessage": "Current dimension size %s is not in the range of %s–%s specified by --input_shape.",
      "Arglist": "cur_dim,shape_range_left,shape_range_right",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Set the dimension size according to --input_shape."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10051",
      "ErrMessage": "Value %s for parameter --job_id exceeds the allowed maximum %s.",
      "Arglist": "id, length",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid argument."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10052",
      "ErrMessage": "AIPP configuration is invalid. Reason: %s.",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10054",
      "ErrMessage": "The required parameter %s for ATC is empty. Another possible reason is that the values of some parameters are not enclosed by quotation marks (\"\").",
      "Arglist": "parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the command line parameter format is correct."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Not_Supported",
      "ErrCode": "E10055",
      "ErrMessage": "The operation is not supported. Reason: %s.",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the configured parameter or used function is supported."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10056",
      "ErrMessage": "Cannot configure both parameters %s and %s simultaneously.",
      "Arglist": "parameter1, parameter2",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "The two parameters cannot be set at the same time. View the parameter descriptions by referring to the documentation."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10057",
      "ErrMessage": "--%s and --%s can only be used together",
      "Arglist": "parameter0, parameter1",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "If the value of --mode is 6, it only needs to be used with --om. Please check and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10058",
      "ErrMessage": "Parameter %s is not configured or is empty.",
      "Arglist": "parameter",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10059",
      "ErrMessage": "%s failed. Reason: %s.",
      "Arglist": "stage, reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10060",
      "ErrMessage": "Parameter is invalid. Reason: %s.",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10061",
      "ErrMessage": "Value %s for parameter %s is invalid. Expected value: %s.",
      "Arglist": "value, parameter, expected_value",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Input_Count",
      "ErrCode": "E10401",
      "ErrMessage": "The number of operator inputs %s exceeds the allowed maximum %s.",
      "Arglist": "expect_num, input_num",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid number of inputs."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Input_Buffer",
      "ErrCode": "E10402",
      "ErrMessage": "Input indexed %s requires a %s buffer, but %s aligned buffer is allocated.",
      "Arglist": "index, expect_size, input_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the data type, dimensions, and shape are correctly set. For details, see the aclGetTensorDescSize API description in API Reference."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Output_Count",
      "ErrCode": "E10403",
      "ErrMessage": "The number of operator outputs %s exceeds the allowed maximum %s.",
      "Arglist": "expect_num, input_num",
      "suggestion": {
        "Possible Cause": "The number of outputs configured for operator execution does not match that described in the operator specifications.",
        "Solution": "Check whether the number of elements in numoutputs is correctly set. The aclopCompile, aclopExecuteV2, and aclopCompileAndExecute APIs may be involved. For details, see API Reference."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Output_Buffer",
      "ErrCode": "E10404",
      "ErrMessage": "Output indexed %s requires a %s buffer, but %s aligned buffer is allocated.",
      "Arglist": "index, expect_size, input_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the data type, dimensions, and shape are correctly set. For details, see the aclGetTensorDescSize API description in API Reference."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10405",
      "ErrMessage": "The number of input buffers is %s, which does not match the number of input tensors %s.",
      "Arglist": "input_num, input_desc_num",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the number of elements in inputDesc and inputs of the operator is correctly set. The aclopExecuteV2 and aclopCompileAndExecute APIs may be involved. For details, see API Reference."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E10406",
      "ErrMessage": "The number of output buffers is %s, which does not match the number of output tensors %s.",
      "Arglist": "out_num, out_desc_num",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the number of elements in outputDesc and outputs of the operator is correctly set. The aclopExecuteV2 and aclopCompileAndExecute APIs may be involved. For details, see API Reference."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_File_Not_Exist",
      "ErrCode": "E10410",
      "ErrMessage": "File %s does not exist.",
      "Arglist": "cfgpath",
      "suggestion": {
        "Possible Cause": "The file specified by the --keep_dtype or --compress_weight_conf argument does not exist.",
        "Solution": "Try again with a valid file directory."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Not_Supported_Operator",
      "ErrCode": "E10501",
      "ErrMessage": "IR for Op %s with optype %s is not registered.",
      "Arglist": "opname,optype",
      "suggestion": {
        "Possible Cause": "1. The environment variable ASCEND_OPP_PATH is not configured. 2. IR is not registered.",
        "Solution": "1. Check whether ASCEND_OPP_PATH is correctly set. 2. Check whether the operator prototype has been registered. For details, see the operator developer guide."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11001",
      "ErrMessage": "input_dim and input_shape are mutually exclusive in NetParameter for Caffe model conversion.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Remove either --input_dim or --input_shape from your atc command line."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11003",
      "ErrMessage": "The number of input_dim fields in the model is %s, which is not 4x the input count %s.",
      "Arglist": "input_dim_size,input_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Modify your Caffe model and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11004",
      "ErrMessage": "The number of input shapes is %s, which does not match the number of inputs %s.",
      "Arglist": "input_shape_size,input_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Modify your Caffe model and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Input_Shape",
      "ErrCode": "E11005",
      "ErrMessage": "The shape is not defined by using --input_shape for input %s.",
      "Arglist": "input",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Modify your Caffe model, or add the input shape to the --input_shape argument in your atc command line."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11008",
      "ErrMessage": "Optype DetectionOutput is unsupported.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Modify your Caffe model and replace DetectionOutput operators with FSRDetectionOutput or SSDDetectionOutput."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Not_Supported_Operator",
      "ErrCode": "E11009",
      "ErrMessage": "No Caffe parser is registered for Op %s with optype %s.",
      "Arglist": "opname,optype",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the Caffe plugin of the operator has been registered."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11012",
      "ErrMessage": "Unknown bottom blob %s at layer %s. The bottom blob is indexed %s.",
      "Arglist": "bottom_blob,layer,bottom_index",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Modify your Caffe model and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11014",
      "ErrMessage": "Failed to find the top blob for layer %s.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "The top blob has no corresponding node in the source Caffe model.",
        "Solution": "Modify your Caffe model and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11015",
      "ErrMessage": "Failed to find the bottom blob for layer %s.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "The bottom blob has no corresponding node in the source Caffe model.",
        "Solution": "Modify your Caffe model and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E11016",
      "ErrMessage": "Failed to add Op %s to NetOutput. Op output index %s is not less than %s. NetOutput input_index %s is not less than %s.",
      "Arglist": "opname,outputindex,totlaloutputindex,inputindex,totlalinputindex",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid --out_nodes argument."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E11017",
      "ErrMessage": "Failed to find node %s specified by --out_nodes.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid --out_nodes argument."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11018",
      "ErrMessage": "Op name %s contains invalid characters.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Allowed characters include: letters, digits, hyphens (-), periods (.), underscores (_), and slashes (/). Modify the Op name and try again."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11021",
      "ErrMessage": "Model file %s contains \"layers\" structures, which have been deprecated in Caffe and unsupported by ATC.",
      "Arglist": "realpath",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Replace layers with layer."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11022",
      "ErrMessage": "Invalid prototxt file.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "No layer structures are found in the Caffe model.",
        "Solution": "Invalid Caffe model. Modify the Caffe model and add the layer field."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11023",
      "ErrMessage": "Weight file contains \"layers\" structures, which have been deprecated in Caffe and unsupported by ATC.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Replace layers with layer."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11024",
      "ErrMessage": "Invalid Caffe weight file.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "No layer structures are found in the Caffe weight file.",
        "Solution": "Invalid Caffe weight file. Modify the Caffe weight file and add the layer field."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11027",
      "ErrMessage": "Op %s with optype %s in the Caffe model has an input node with shape size 0.",
      "Arglist": "layername,layertype",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Invalid Caffe model. Modify the input shape of the node."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11029",
      "ErrMessage": "Op %s exists in model file but not found in weight file.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid Caffe model or weight file. Ensure that the two files match with each other."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11032",
      "ErrMessage": "Failed to parse message %s. The error field is %s. Reason: %s.",
      "Arglist": "message_type,name,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the Caffe model supports the field."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Weight",
      "ErrCode": "E11033",
      "ErrMessage": "Failed to convert the weight file. Blob %s of size %s is invalid. Reason: %s.",
      "Arglist": "opname,blobsize,reason",
      "suggestion": {
        "Possible Cause": "The blob size of the node in the Caffe weight file does not match the number of elements calculated based on its shape.",
        "Solution": "Try again with a valid Caffe model or weight file. Ensure that the two files match with each other."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11035",
      "ErrMessage": "The top size of data node %s is not 1 but %s.",
      "Arglist": "opname,size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Invalid Caffe model. Change the number of outputs for the data node to 1."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11036",
      "ErrMessage": "Data nodes have duplicate top blobs %s.",
      "Arglist": "topname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Invalid Caffe model. Make sure the data node has a unique output name."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Caffe_Model_Data",
      "ErrCode": "E11037",
      "ErrMessage": "Op %s has zero outputs.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Nodes in the Caffe model must have at least one output."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Input_Index",
      "ErrCode": "E12004",
      "ErrMessage": "Failed to register the prototype of Op %s. If input index is less than 0, then input index –%s (absolute value) must be less than the input count %s",
      "Arglist": "opname,inputIdx,inputsize",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "When the Const input is converted to an attribute, check whether the input index is correctly set."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_TensorFlow_Model_Data",
      "ErrCode": "E12009",
      "ErrMessage": "Input %s for Op %s is not found in graph_def.",
      "Arglist": "opname,inputopname",
      "suggestion": {
        "Possible Cause": "The input name of the node is not found in the graph.",
        "Solution": "Try again with a valid TensorFlow model."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_TensorFlow_Model_Data",
      "ErrCode": "E12013",
      "ErrMessage": "Failed to find a subgraph by the name %s.",
      "Arglist": "functionname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. To use function subgraphs to convert a TensorFlow model, place the subgraph .proto description file in the same directory as the model file and name it graph_def_library.pbtxt. 2. Run the func2graph.py script in the ATC installation directory to save the subgraphs to graph_def_library.pbtxt."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_TensorFlow_Model_Data",
      "ErrCode": "E12029",
      "ErrMessage": "Failed to find the subgraph library.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "The model to convert contains function subgraphs, but the graph_def_library.pbtxt file is not found.",
        "Solution": "1. To use function subgraphs to convert a TensorFlow model, place the subgraph .proto description file in the same directory as the model file and name it graph_def_library.pbtxt. 2. Run the func2graph.py script in the ATC installation directory to save the subgraphs to graph_def_library.pbtxt."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Invalid_Path",
      "ErrCode": "E13000",
      "ErrMessage": "Path %s is empty. Reason: %s.",
      "Arglist": "path,errmsg",
      "suggestion": {
        "Possible Cause": "The file does not exist.",
        "Solution": "Try again with a valid directory."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Open",
      "ErrCode": "E13001",
      "ErrMessage": "Failed to open file %s. Reason: %s.",
      "Arglist": "file,errmsg",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Fix the error according to the error message."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Directory_Operation_Error_Name_Too_Long",
      "ErrCode": "E13002",
      "ErrMessage": "Directory %s is too long. Keep the length within %s characters.",
      "Arglist": "filepath,size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid file directory."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Read",
      "ErrCode": "E13003",
      "ErrMessage": "Failed to read file %s. Reason: %s.",
      "Arglist": "file,errmsg",
      "suggestion": {
        "Possible Cause": "Failed to read the file.",
        "Solution": "Fix the error according to the error message."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Write",
      "ErrCode": "E13004",
      "ErrMessage": "Failed to write file %s. Reason: %s.",
      "Arglist": "file,errmsg",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Parse",
      "ErrCode": "E13005",
      "ErrMessage": "Failed to parse file %s.",
      "Arglist": "file",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check that a matched Protobuf version is installed and try again with a valid file. For details, see section \"--framework\" in ATC Instructions."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Parse",
      "ErrCode": "E13006",
      "ErrMessage": "Type %s of file %s is incorrect. Expected type: %s.",
      "Arglist": "file,current_type,expect_type",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Name",
      "ErrCode": "E13009",
      "ErrMessage": "Operator %s already exists in the graph. Ensure that operator names are unique.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Ensure that the operators in the graph have unique names."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Not_Supported_Operator",
      "ErrCode": "E13010",
      "ErrMessage": "No operator plugin is registered for Op: %s, optype: %s.",
      "Arglist": "opname,optype",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. If the operator is a custom operator, register related deliverables.\n2. If the operator is a built-in operator, install the package that supports this operator version."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E13014",
      "ErrMessage": "Value %s of parameter %s for Op %s is invalid. Reason: %s.",
      "Arglist": "value,parameter,opname,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Invalid_File_Size",
      "ErrCode": "E13015",
      "ErrMessage": "File %s has a size of %s, which is out of valid range (0, %s].",
      "Arglist": "file,size,maxsize",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Parse",
      "ErrCode": "E13018",
      "ErrMessage": "Failed to parse file %s through google::protobuf::TextFormat::Parse.",
      "Arglist": "protofile",
      "suggestion": {
        "Possible Cause": "The file may not be in valid Protobuf format.",
        "Solution": "Check whether the Protobuf file is valid."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_OM_Model_Size",
      "ErrCode": "E13023",
      "ErrMessage": "Model %s has a size of %s bytes, which exceeds system limit of %s bytes.",
      "Arglist": "item,size,maxsize",
      "suggestion": {
        "Possible Cause": "The generated OM model is too large and therefore cannot be dumped to the disk.",
        "Solution": "Reduce the model size."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Config_Error_Invalid_Environment_Variable",
      "ErrCode": "E13024",
      "ErrMessage": "Value %s for environment variable %s is invalid. Reason: %s.",
      "Arglist": "value,env,situation",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Reset the environment variable by referring to the setup guide."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E13025",
      "ErrMessage": "Input tensor is invalid. Reason: %s.",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "File_Operation_Error_Invalid_Path",
      "ErrCode": "E13026",
      "ErrMessage": "Input path %s is invalid. Reason: %s",
      "Arglist": "pathname,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Network_Connection",
      "ErrCode": "E13027",
      "ErrMessage": "Failed to connect to the peer address %s.",
      "Arglist": "address",
      "suggestion": {
        "Possible Cause": "The ipaddr, port or token is invalid.",
        "Solution": "Check the ipaddr, port or token in the configuration file, and ensure that the configuration is correct."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Compilation_Error_Execute_Custom_Fusion_Pass",
      "ErrCode": "E13028",
      "ErrMessage": "Failed to run custom fusion pass %s. Return code: %s. Reason: %s.",
      "Arglist": "passname,retcode,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check the error log for details and verify whether the fusion logic is correct."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Compilation_Error_Load_Custom_Fusion_Pass",
      "ErrCode": "E13029",
      "ErrMessage": "Failed to load custom fusion pass lib %s. Reason: %s.",
      "Arglist": "passlibname,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Analyze the failure reason mentioned above. Below are some typical solutions for common dlopen failures:\n1. Verify that the library path is correct and the file exists.\n2. Ensure the library and its dependencies have the correct permissions.\n3. Check that all dependencies are available using the 'ldd' command."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Initialization_Error_Register_Custom_Fusion_Pass",
      "ErrCode": "E13030",
      "ErrMessage": "Failed to get custom fusion pass func %s. Reason: %s.",
      "Arglist": "passname,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check that custom pass registration is valid."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E13031",
      "ErrMessage": "Output tensor is invalid. Reason: %s.",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Operator_Compilation_Parameter",
      "ErrCode": "E14001",
      "ErrMessage": "Argument %s for Op %s with optype %s is invalid. Reason: %s.",
      "Arglist": "value,opname,optype,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the type, input, and output of the operator match the configured parameters."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_Tensor_Attribute",
      "ErrCode": "E14002",
      "ErrMessage": "In the current process, the attribute of %s must be obtained successfully. Reason: %s.",
      "Arglist": "attribute,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "The attribute in the error message must be set for the operator."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_ONNX_Model_Data",
      "ErrCode": "E16001",
      "ErrMessage": "The model has no %s node.",
      "Arglist": "value",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the ONNX model contains the input node."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_ONNX_Model_Data",
      "ErrCode": "E16002",
      "ErrMessage": "No ONNX parser is registered for optype %s.",
      "Arglist": "optype",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check the version of the installation package and reinstall the package. For details, see the installation reference."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_ONNX_Model_Data",
      "ErrCode": "E16004",
      "ErrMessage": "ONNX model has no graph.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Invalid ONNX model. Modify the ONNX model and add the graph information."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument_ONNX_Model_Data",
      "ErrCode": "E16005",
      "ErrMessage": "The model has %s --domain_version fields, but only one is allowed.",
      "Arglist": "domain_version_size",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Invalid ONNX model. Modify the ONNX model. If no domain is specified on the operator node, only one domain can be specified on the model."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Inner_Error",
      "ErrCode": "E19999",
      "ErrMessage": "Unknown error occurred. Please check the log.",
      "Arglist": "",
      "suggestion": {
        "Possible Cause": "System terminated abnormally without valid error messages.",
        "Solution": "In this scenario, collect the logs generated when the fault occurs and locate the fault based on the logs."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Performance_Not_Optimal_Operator",
      "ErrCode": "W11001",
      "ErrMessage": "Op %s does not hit the high-priority operator information library, which might result in compromised performance.",
      "Arglist": "opname",
      "suggestion": {
        "Possible Cause": "The operator does not hit the high-priority operator information library, which might result in compromised performance.",
        "Solution": "Submit an issue to request for support at https://gitcode.com/cann."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Config_Error_Weight_Configuration",
      "ErrCode": "W11002",
      "ErrMessage": "In the compression weight configuration file %s, some nodes do not exist in graph: %s.",
      "Arglist": "filename, opnames",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the weight file matches the model file."
      }
    },
    {
      "errClass": "GE Errors",
      "errTitle": "Config_Error_Operator_Missing_Implementation",
      "ErrCode": "W11003",
      "ErrMessage": "Operator %s lacks required %s implementation.",
      "Arglist": "optype, func",
      "suggestion": {
        "Possible Cause": "Incomplete operator implementation.",
        "Solution": "Ensure all operator required implementations(e.g., tiling) are provided. See the operator developer guide for details."
      }
    }
  ]
}
)";

REG_FORMAT_ERROR_MSG(ge_error_code.c_str(), ge_error_code.size());
} // namespace