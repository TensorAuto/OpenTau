ROS Conversion
==============

This tutorial explains how to convert ROS bags to LeRobot dataset format.

The ROS to LeRobot conversion script is located in ``src/opentau/scripts/convert_ros_to_lerobot.py`` which takes a config file as input. Example config is mentioned in ``configs/examples/ros2lerobot.json``.

The example config looks like:

.. code-block:: javascript

    {
    "input_path": <path to ros bag folder>,
    "output_path": <path to lerobot dataset output folder>,
    "fps": <frames per second for the output dataset. The input dataset can be at high frequency like 1000 fps, but the output dataset is at 30 fps>,
    "joint_order": <list of joint names in the order of the state vector>,
    "dataset_features": {
        <feature_name in lerobot dataset>: {
                "ros_topic": <ros topic name stored in rosbag>,
                "topic_attribute":<attribute name stored in ros topic>,
                "enum_values": <enum vales stored in ros topic. Used to parse the field and apply appropriate preprocessing>,
                "dtype":<dtype of the feature in lerobot dataset>,
                "shape":<shape of the feature in lerobot dataset>
            },
        }
    }

The following structure for ros bags should be followed:

<input_path>/
    <bag_name>/
        recording/
            - <bag_file>.mcap
            - metadata.yaml
    <other_bag_name>/
        recording/
            - <bag_file>.mcap
            - metadata.yaml

The metadata.yaml file should contain the task name for the dataset else "task not defined" will be used as task name/

The script will process all the bag files in the input path and create a LeRobot dataset in the output path. Each ros bag will be converted to a single episode in the dataset.


Each features should have enum values to parse the field and apply appropriate preprocessing. The function to parse the field is located in ``src/opentau/utils/ros2lerobot.py``.

Example of enum values for a feature:

.. code-block:: python

    class ImageExtractor(FeatureExtractor):
    def __call__(self, msg: Any, ros_topic: str, attribute: str) -> Any:
        try:
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(msg.data))
            # Convert to numpy array
            image_np = np.array(image)
            # Handle RGBA if necessary, or just ensure RGB
            if image_np.shape[-1] == 4:
                image_np = image_np[..., :3]
            return image_np

        except (KeyError, AttributeError, TypeError, Exception) as e:
            logging.warning(f"Error extracting {attribute} from {ros_topic}: {e}")
            return None

The class should be mapped to the enum values in the config file. and also mapped in the extractors dictionary.

.. code-block:: python

    EXTRACTORS = {
        "image": ImageExtractor,
    }


The scripts only supports video dtype for image features.
