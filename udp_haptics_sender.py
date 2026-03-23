import argparse
import math
import socket
import time


def clamp01(value):
    return max(0.0, min(1.0, value))


def send_packet(sock, host, port, left, right):
    payload = f"{clamp01(left):.4f} {clamp01(right):.4f}".encode("utf-8")
    sock.sendto(payload, (host, port))


def triangle_wave(phase):
    wrapped = phase % 1.0
    if wrapped < 0.25:
        return 0.5 + 2.0 * wrapped
    if wrapped < 0.75:
        return 1.0 - 2.0 * (wrapped - 0.25)
    return 2.0 * (wrapped - 0.75)


def wave_value(mode, elapsed, frequency, minimum, maximum, phase):
    if mode == "static":
        return clamp01(maximum)

    shifted_time = elapsed + phase / max(frequency, 1e-6)
    cycle_phase = shifted_time * frequency

    if mode == "sine":
        normalized = 0.5 + 0.5 * math.sin(2.0 * math.pi * cycle_phase)
    elif mode == "triangle":
        normalized = triangle_wave(cycle_phase)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return clamp01(minimum + (maximum - minimum) * normalized)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Send UDP haptics force values to the Oculus app. "
            "Payload format is: left right"
        )
    )
    parser.add_argument("host", help="Headset IP address")
    parser.add_argument("--port", type=int, default=9000, help="UDP port on the headset")
    parser.add_argument("--left", type=float, default=0.0, help="Static left controller force in [0, 1]")
    parser.add_argument("--right", type=float, default=0.0, help="Static right controller force in [0, 1]")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="If > 0, resend continuously at this frequency in Hz until interrupted",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "sine", "triangle"],
        default="static",
        help="Waveform used for continuous sending",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1.0,
        help="Wave frequency in Hz for sine/triangle mode",
    )
    parser.add_argument(
        "--min-force",
        type=float,
        default=0.0,
        help="Minimum force used by both channels in dynamic mode",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=1.0,
        help="Maximum force used by both channels in dynamic mode",
    )
    parser.add_argument(
        "--left-scale",
        type=float,
        default=1.0,
        help="Additional scale for the left channel after waveform generation",
    )
    parser.add_argument(
        "--right-scale",
        type=float,
        default=1.0,
        help="Additional scale for the right channel after waveform generation",
    )
    parser.add_argument(
        "--left-phase",
        type=float,
        default=0.0,
        help="Left channel phase offset in cycles for dynamic mode",
    )
    parser.add_argument(
        "--right-phase",
        type=float,
        default=0.0,
        help="Right channel phase offset in cycles for dynamic mode",
    )
    args = parser.parse_args()

    args.min_force = clamp01(args.min_force)
    args.max_force = clamp01(args.max_force)
    if args.max_force < args.min_force:
        args.min_force, args.max_force = args.max_force, args.min_force

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        if args.rate <= 0.0:
            send_packet(sock, args.host, args.port, args.left, args.right)
            return

        interval = 1.0 / args.rate
        start_time = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - start_time
                if args.mode == "static":
                    left = args.left
                    right = args.right
                else:
                    base_left = wave_value(
                        args.mode,
                        elapsed,
                        args.frequency,
                        args.min_force,
                        args.max_force,
                        args.left_phase,
                    )
                    base_right = wave_value(
                        args.mode,
                        elapsed,
                        args.frequency,
                        args.min_force,
                        args.max_force,
                        args.right_phase,
                    )
                    left = base_left * args.left_scale
                    right = base_right * args.right_scale

                send_packet(sock, args.host, args.port, left, right)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()

