Until ORT is release with Xamarin support in the nuget package you need to
  - get the managed and native nuget packages from the internal Zip-Nuget-Java packaging pipeline for a build of master
  - put that in a local directory
  - update the nuget.config to point to that directory

Additionally, the fastrcnn model is required to be in the Models directory.

From this directory:
> mkdir Models
> wget https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx