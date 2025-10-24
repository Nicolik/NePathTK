import os
import sys

from definitions import OPENSLIDE

os.environ['PATH'] = OPENSLIDE + ";" + os.environ['PATH']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def check_openslide():
    print("🔍 Checking OpenSlide...")
    try:
        import openslide
        ver = openslide.__version__
        print(f"✅ OpenSlide Python module found (version {ver})")
    except ImportError:
        print("❌ OpenSlide Python module not installed. Install with: pip install openslide-python")
        return False

    # Check if the OPENSLIDE env var is set in definitions.py (or globally)
    try:
        import definitions
        openslide_path = getattr(definitions, "OPENSLIDE", None)
        if openslide_path and os.path.exists(openslide_path):
            print(f"✅ OPENSLIDE path set to: {openslide_path}")
        else:
            print("⚠️ OPENSLIDE path not set or invalid in definitions.py")
    except ImportError:
        print("⚠️ definitions.py not found. Make sure you copied definitions.py.base and configured it.")
    return True


def check_paquo_qupath():
    print("\n🔍 Checking QuPath / Paquo...")
    try:
        import paquo
        from paquo.projects import QuPathProject

        # Check .paquo.toml exists
        if not os.path.exists(".paquo.toml"):
            print("❌ .paquo.toml not found. Copy .paquo.toml.base and set the QuPath directory.")
            return False
        else:
            print(f"✅ .paquo.toml found")

        # Try reading config
        from paquo import _config
        print("keys:", _config.settings.keys())
        qp_path = _config.settings['QUPATH_DIR']
        if qp_path and os.path.exists(qp_path):
            print(f"✅ QuPath found at: {qp_path}")
        else:
            print("⚠️ QuPath path not configured or invalid in .paquo.toml")
            return False

        print("✅ Paquo can interface with QuPath")
        return True

    except ImportError:
        print("❌ Paquo not installed. Install with: pip install paquo")
        return False


if __name__ == "__main__":
    ok1 = check_openslide()
    ok2 = check_paquo_qupath()

    if ok1 and ok2:
        print("\n🎉 Setup verification successful: NePathTK environment looks good!")
        sys.exit(0)
    else:
        print("\n⚠️ Setup verification failed: please fix the issues above.")
        sys.exit(1)
