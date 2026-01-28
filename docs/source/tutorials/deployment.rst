Deployment
==========

OpenTau supports deploying trained models as gRPC inference servers for robot control. This allows you to run model inference on a GPU server while the robot communicates with it over the network using gRPC.

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Overview
--------

The deployment system consists of two main components:

1. **gRPC Server**: Runs on a machine with GPU access, loads a trained policy model, and serves inference requests.
2. **gRPC Client**: Runs on the robot (typically with ROS 2), sends observations to the server, and receives action predictions.

The server and client communicate using Protocol Buffers (protobuf) over gRPC, allowing for efficient serialization and network communication.

Setting up the Server
---------------------

The gRPC server loads a trained policy model and serves inference requests. To set up the server, you need:

1. A trained model checkpoint (containing ``train_config.json`` and ``model.safetensors``)
2. A configuration file with server settings
3. GPU access on the server machine

Configuration
~~~~~~~~~~~~~

The server configuration is part of the training pipeline configuration. You can add a ``server`` section to your config file or override it via command-line arguments.

Example configuration file with server settings:

.. code-block:: javascript

    {
        "policy": {
            "type": "pi05",
            "pretrained_path": "outputs/train/pi05/checkpoints/000040",
            ...
        },
        "server": {
            "port": 50051,
            "max_workers": 4,
            "max_send_message_length_mb": 100,
            "max_receive_message_length_mb": 100
        },
        "resolution": [224, 224],
        "num_cams": 2,
        "max_state_dim": 32,
        "max_action_dim": 32,
        ...
    }


Running the Server
~~~~~~~~~~~~~~~~~~

To start the gRPC server, use the server script:

.. code-block:: bash

    python src/opentau/scripts/grpc/server.py --config_path=/path/to/config.json

You can override server settings via command-line arguments:

.. code-block:: bash

    python src/opentau/scripts/grpc/server.py \
        --config_path=/path/to/config.json \
        --server.port=50051 \
        --server.max_workers=8

The server will:

1. Load the policy model from the checkpoint
2. Move it to the available device (GPU if available, otherwise CPU)
3. Set the model to evaluation mode
4. Start listening on the specified port

Once started, you should see output like:

.. code-block:: text

    Server started on port 50051
    Policy: pi05
    Device: cuda:0
    Max workers: 4
    Waiting for requests...

Health Check
~~~~~~~~~~~~

The server provides a health check endpoint that you can use to verify it's running correctly. The client can call this endpoint to check:

- Server health status
- Model name
- Device information
- GPU memory usage (if GPU is available)

Setting up the Client
---------------------

The gRPC client runs on the robot and communicates with the server. The client implementation includes ROS 2 integration for subscribing to robot state and publishing motor commands.

.. note::
   The client script provided (``src/opentau/scripts/grpc/client.py``) is an example implementation and is intended as a starting point. You will need to edit it to match your specific ROS 2 environment, topics, message types, and robot interfaces. Review and adapt the code to your robot setup before deploying.


Prerequisites
~~~~~~~~~~~~~

The client requires:

- ROS 2 installed and configured
- Python packages: ``grpcio``, ``grpcio-tools``, ``rclpy``
- Access to ROS 2 message types (e.g., ``sensor_msgs/JointState``, custom motor command messages)

Running the Client
~~~~~~~~~~~~~~~~~~

To run the ROS 2 client:

.. code-block:: bash

    python src/opentau/scripts/grpc/client.py \
        --server_address 192.168.1.100:50051 \
        --prompt "pick up the red block" \
        --control_frequency 30.0 \
        --num_cameras 2 \
        --timeout 30.0

Client arguments:

- ``--server_address``: Server address in format ``host:port`` (default: ``localhost:50051``)
- ``--prompt``: Language instruction for the policy (required)
- ``--control_frequency``: Control loop frequency in Hz (default: 30.0)
- ``--num_cameras``: Number of camera images to send (default: 2)
- ``--timeout``: gRPC timeout in seconds (default: 30.0)

The client will:

1. Connect to the gRPC server
2. Subscribe to ``/joint_states`` for robot state
3. Create camera images (or subscribe to camera topics in a custom implementation)
4. Send observations to the server at the specified control frequency
5. Publish motor commands to ``/motor_command_controller/motor_commands``


Protocol Buffer Generation
--------------------------

The gRPC communication uses Protocol Buffers defined in ``robot_inference.proto``. If you modify the proto file, you need to regenerate the Python code.

To regenerate the protobuf code:

.. code-block:: bash

    cd /path/to/OpenTau
    ./src/opentau/scripts/grpc/generate_proto.sh

This script:

1. Generates ``robot_inference_pb2.py`` and ``robot_inference_pb2_grpc.py`` from the proto file
2. Fixes import paths to work with the package structure

The generated files are automatically included in the package and should not be manually edited.

gRPC Service API
----------------

The server implements the ``RobotPolicyService`` with three RPC methods:

GetActionChunk
~~~~~~~~~~~~~~~

Single request-response RPC for getting an action chunk from observations:

.. code-block:: python

    request = ObservationRequest(
        images=[camera_image_1, camera_image_2, ...],
        robot_state=RobotState(state=[...]),
        prompt="pick up the red block",
        request_id="req_1",
        timestamp_ns=time.time_ns()
    )
    response = stub.GetActionChunk(request)

The response contains:

- ``action_chunk``: List of action vectors (one per timestep in the chunk)
- ``timestamp_ns``: Server timestamp
- ``request_id``: Matching request ID
- ``inference_time_ms``: Time taken for inference

StreamActionChunks
~~~~~~~~~~~~~~~~~~

Streaming RPC for continuous inference. The robot sends a stream of observations, and the server responds with a stream of action chunks:

.. code-block:: python

    def observation_stream():
        while True:
            yield create_observation_request(...)

    for response in stub.StreamActionChunks(observation_stream()):
        process_action_chunk(response.action_chunk)

HealthCheck
~~~~~~~~~~~

Health check endpoint to verify server status:

.. code-block:: python

    response = stub.HealthCheck(HealthCheckRequest())
    # response contains: healthy, status, model_name, device, gpu_memory_used_gb, gpu_memory_total_gb


Troubleshooting
---------------

Connection Issues
~~~~~~~~~~~~~~~~~

If the client cannot connect to the server:

- Verify the server is running: Check server logs and ensure it's listening on the correct port
- Check network connectivity: Use ``ping`` or ``telnet`` to verify the server is reachable
- Check firewall settings: Ensure the server port is not blocked
- Verify the server address: Use the correct IP address and port

Timeout Errors
~~~~~~~~~~~~~~

If requests timeout:

- Increase the timeout value: Use ``--timeout`` argument or increase ``timeout_seconds`` in the client config
- Check server performance: Monitor GPU usage and inference times
- Reduce image size: Use lower resolution images or better compression
- Check network latency: Ensure low latency between robot and server

Image Encoding Issues
~~~~~~~~~~~~~~~~~~~~~

If images are not decoded correctly:

- Verify image encoding: Ensure the client and server use compatible encodings (JPEG, PNG, or raw)
- Check image format: Ensure images are RGB format with correct dimensions
- Verify resolution: Ensure images match the expected resolution in the config

Performance Optimization
------------------------

To improve server performance:

1. **Increase max_workers**: For handling more concurrent requests:

   .. code-block:: bash

       --server.max_workers=8

2. **Use GPU**: Ensure the server has GPU access and CUDA is properly configured

3. **Optimize model**: The server automatically uses ``torch.compile`` if available for faster inference

4. **Adjust message sizes**: Increase message length limits if sending large images:

   .. code-block:: javascript

       "server": {
           "max_send_message_length_mb": 200,
           "max_receive_message_length_mb": 200
       }

To improve client performance:

1. **Adjust control frequency**: Match the frequency to your robot's capabilities and network latency
2. **Use image compression**: Use JPEG encoding with appropriate quality settings
3. **Batch requests**: If using streaming, ensure continuous observation flow

Example Deployment Workflow
----------------------------

1. **Train a model**: Train your policy model using the training pipeline

2. **Prepare server config**: Create or modify a config file with server settings

3. **Start the server**:

   .. code-block:: bash

       python src/opentau/scripts/grpc/server.py --config_path=deployment_config.json

4. **Test connection**: Use the health check or a simple test client to verify the server is responding

5. **Deploy client on robot**: Copy the client script to the robot and configure it with the server address

6. **Run the client**: Start the ROS 2 client with appropriate arguments

7. **Monitor**: Check server logs and client logs for any issues

For more information on training models, see the :doc:`/tutorials/training` guide. For inference without gRPC, see the :doc:`/tutorials/inference` guide.
