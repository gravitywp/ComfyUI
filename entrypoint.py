import shutil
import json
import sys
import uuid
import comfy.options
# comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
from moviepy.editor import *

from comfy.cli_args import parser
from comfy import cli_args

args_list = sys.argv[1:]
if "FC_CUSTOM_CONTAINER_EVENT" in os.environ:
    event = json.loads(os.environ.get('FC_CUSTOM_CONTAINER_EVENT'))
    for k, v in event.items():
        args_list.append("--%s" % k)
        args_list.append(v)
print("args............  ", " ".join(args_list))
args = parser.parse_args(args_list)
cli_args.args = args

def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            print(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        print("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

execute_prestartup_script()


# Main code
import asyncio
import itertools
import shutil
import threading
import gc


if os.name == "nt":
    import logging
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        print("Set cuda device to:", args.cuda_device)

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

import comfy.utils
import yaml

import execution
import server
from server import BinaryEventTypes, PromptServer
from nodes import init_custom_nodes
import comfy.model_management

def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            print("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")

def prompt_worker(q, server):
    e = execution.PromptExecutor(server)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = None
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id, e.outputs_ui)
            if server.client_id is not None:
                server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            print("Prompt executed in {:.2f} seconds".format(execution_time))

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False

async def run(server: PromptServer, address='', port=8188, verbose=True, call_on_start=None, prompt_id=str):
    await asyncio.gather(server.start(address, port, verbose, call_on_start))
    while True:
        await asyncio.sleep(3)
        queue_info = server.get_queue_info()
        if queue_info['exec_info']['queue_remaining'] == 0:
            return
        else:
            print(queue_info)


def hijack_progress(server):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        server.send_sync("progress", {"value": value, "max": total}, server.client_id)
        if preview_image is not None:
            server.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server.client_id)
    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                print("Adding extra search path", x, full_path)
                folder_paths.add_model_folder_path(x, full_path)


if __name__ == "__main__":
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        print(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = server.PromptServer(loop)
    q = execution.PromptQueue(server)

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    init_custom_nodes()

    cuda_malloc_warning()

    server.add_routes()
    hijack_progress(server)

    threading.Thread(target=prompt_worker, daemon=True, args=(q, server,)).start()

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        print(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    #These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        print(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.quick_test_for_ci:
        exit(0)

    # call_on_start = None
    # if args.auto_launch:
    #     def startup_server(address, port):
    #         import webbrowser
    #         if os.name == 'nt' and address == '0.0.0.0':
    #             address = '127.0.0.1'
    #         webbrowser.open(f"http://{address}:{port}")
    #     call_on_start = startup_server


    if os.path.isfile(args.input_video):
        if os.path.exists("./input/input.mp4"):
            os.remove(os.path.join("./input", "input.mp4"))
        shutil.copyfile(args.input_video, os.path.join("./input", "input.mp4"))
    else:
        raise Exception("not found input file {}".format(args.input_video))


    data = json.load(open(args.workflow_json, mode="rb"))
    frames_id = data["info"]["frames"]
    prompt_text_id = data["info"]["prompt"]
    video_name_id = data["info"]["video_name"]
    model_name_id = data['info']['model']
    prompt = data["prompt"]

    prompt[model_name_id]['inputs']['ckpt_name'] = args.model_name
    prompt[frames_id]['inputs']['value'] = args.frames
    prompt[video_name_id]['inputs']['video'] = "input.mp4"
    prompt[prompt_text_id]['inputs']['text'] += args.prompt_text

    print("To generate {} frames.".format(args.frames))

    valid = execution.validate_prompt(prompt)
    extra_data = {}
    if valid[0]:
        prompt_id = str(uuid.uuid4())
        outputs_to_execute = valid[2]
        server.prompt_queue.put((server.number, prompt_id, prompt, extra_data, outputs_to_execute))
    else:
        raise Exception("Valid prompt failed.")
    
    try:
        loop.run_until_complete(run(server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=None))

        video_names = os.listdir(os.path.join(args.output_directory, "videos"))
        video_names.sort(reverse=True)
        r_video_name = video_names[0]
        if os.path.exists(args.output_video):
            os.remove(args.output_video)
        # 从视频文件中创建 VideoFileClip 对象
        audio_clip = AudioFileClip(args.input_video)
        audio_clip = audio_clip.set_end((0, args.frames / 8))


        # 从音频文件中创建 AudioFileClip 对象
        video_file = os.path.join(args.output_directory, "videos", r_video_name)
        video_clip = VideoFileClip(video_file)

        # 创建一个 CompositeAudioClip 对象，将视频文件的音频和新音频文件合并

        # 将视频文件的音频替换为新的合并后的音频
        video_clip = video_clip.set_audio(audio_clip)

        # 将合并后的视频文件写入新文件
        video_clip.write_videofile(args.output_video)

        print("Output video: ", args.output_video)
    except KeyboardInterrupt:
        print("\nStopped server")

    cleanup_temp()