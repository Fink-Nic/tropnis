# type: ignore
import time
import signal
from resources.tropnis import GammaLoopIntegrand

def main():
    integrand = GammaLoopIntegrand("settings/scalar_box.toml", n_cores=3)

    # ensure default signal handling for main process
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        a = 3
        while a > 0:
            # sleep in small increments so Ctrl+C can interrupt
            time.sleep(3)
            a -= 1
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()

if __name__ == "__main__":
    main()