"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
from src.tetris import Tetris

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--video", type=bool, default=False, help="Record video")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--num_eps", type=int, default=20)

    args = parser.parse_args()
    return args

def tester(w):
    return 1

def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    if opt.video:
        out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    
    all_scores = np.zeros(opt.num_eps)
    for eps_idx in range(opt.num_eps):

        env.reset()
        while True:
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            if torch.cuda.is_available():
                next_states = next_states.cuda()
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            if opt.video:
                _, done = env.step(action, render=False, video=out)
            else:
                _, done = env.step(action, render=False)

            if done:
                if opt.video:
                    out.release()
                break
        
        all_scores[eps_idx] = env.cleared_lines
        if eps_idx > 0:
            print(eps_idx, np.mean(all_scores[:eps_idx]), np.max(all_scores[:eps_idx]))
    
    print(f"Avg: {np.round(np.mean(all_scores), 3)}, \
            Max: {np.max(all_scores)}, \
            Min: {np.min(all_scores)}, \
            STD: {np.std(all_scores)}")


if __name__ == "__main__":
    opt = get_args()
    test(opt)