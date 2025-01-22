import argparse
import threading
import os
import pandas as pd
from subprocess import Popen, PIPE
from test_runner import test_runner
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent Object Detection Wrapper")
    # parser.add_argument('--script', required=True, help="Path to the script to execute")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads to use")
    parser.add_argument('--max_shift', type=float, required=True, help="Maximum shift value")
    parser.add_argument('--stride', type=int, default=1, help="Stride value")
    #parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use")
    parser.add_argument('-r', '--random_weights',
                        dest="random_weights",
                        action="store_true",
                        default=False,
                        help="use random weights")
    parser.add_argument('-fpn', 
                        dest="is_afc_fpn",
                        action="store_true",
                        default=False,
                        help="use afc fpn+rpn config")
    parser.add_argument('--only-fpn', 
                        dest="is_only_fpn",
                        action="store_true",
                        default=False,
                        help="use afc only fpn config")
    parser.add_argument('-is_afc', 
                        dest="is_afc",
                        action="store_true",
                        default=False,
                        help="use afc config")
    parser.add_argument('-is_cyclic', 
                        dest="is_cyclic",
                        action="store_true",
                        default=False,
                        help="use cyclic shift or crop shift")
    return parser.parse_args()


def run_script(args, start_idx, end_idx, thread_idx, output_file):

    print(f"[Thread {thread_idx}] Running")
    try:
        test_runner(is_afc=args.is_afc,
                    max_shift=args.max_shift,
                    stride=args.stride,
                    random_weights=args.random_weights,
                    cyclic_flag=args.is_cyclic,
                    start_index=start_idx,
                    end_index=end_idx,
                    is_afc_fpn=args.is_afc_fpn,
                    gpu_num=thread_idx,
                    csv_filename=output_file,
                    is_only_fpn=args.is_only_fpn)

    except Exception as e:
        print(f"[Thread {thread_idx}] Error: {e}\n")

    return output_file


def main():
    args = parse_args()
    args.output = f"max_shift_{args.max_shift}_stride_{args.stride}"
    if args.is_afc:
        args.output += "_backbone_afc"
    if args.is_afc_fpn:
        args.output += "_full_network_afc"
    if args.is_only_fpn:
        args.output += "_only_fpn_afc"
    if args.random_weights:
        args.output += "_random_weights"
    if args.is_cyclic:
        args.output += f"_cyclic.csv"
    else:
        args.output += f"_crop.csv"

    # Load the image file names to divide the workload
    data_dir = '/home/hagaymi/data/coco'  # Adjust to actual dataset path
    data_type = 'val2017'
    image_dir = os.path.join(data_dir, data_type)
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    ]

    total_files = len(image_files)
    chunk_size = total_files // args.threads

    threads = []
    result_files = []

    for i in range(args.threads):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i != args.threads - 1 else total_files
        output_file = f"max_shift_{args.max_shift}_stride_{args.stride}_results_thread_{i}"
        if args.is_afc:
            output_file += "_backbone_afc"
        if args.is_afc_fpn:
            output_file += "_full_network_afc"
        if args.is_only_fpn:
            output_file += "_only_fpn_afc"
        if args.random_weights:
            output_file += "_random_weights"
        if args.is_cyclic:
            output_file += f"_cyclic.csv"
        else:
            output_file += f"_crop.csv"
        thread = threading.Thread(
            target=run_script,
            args=(args, start_idx, end_idx, i, output_file),
        )
        threads.append(thread)
        result_files.append(output_file)
        thread.start()
        time.sleep(10)
    for thread in threads:
        thread.join()

    # Combine all results into the final CSV
    combined_df = pd.DataFrame()

    for i, result_file in enumerate(result_files):
        if os.path.exists(result_file):
            if i == 0:
                # Read the first file with headers
                combined_df = pd.read_csv(result_file)
            else:
                # Read subsequent files without headers
                df = pd.read_csv(result_file, header=None)
                df.columns = combined_df.columns  # Assign column names from the first file
                combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(args.output, index=False)
    print(f"Combined results written to {args.output}")

    # Clean up temporary files
    for result_file in result_files:
        if os.path.exists(result_file):
            os.remove(result_file)


if __name__ == "__main__":
    main()
