{
  description = "dt-apriltags: Python bindings for the Apriltags library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Build tools
            cmake
            gnumake
            gcc

            # Python and development dependencies
            python3
            python3Packages.pip
            python3Packages.virtualenv

            # OpenCV for building examples
            opencv

            # Development tools
            git
          ];

          shellHook = ''
            # Set up virtual environment
            VENV_DIR=".venv"
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating Python virtual environment..."
              python -m venv "$VENV_DIR"
            fi

            source "$VENV_DIR/bin/activate"

            # Upgrade pip and install test dependencies if not already installed
            if ! pip show build > /dev/null 2>&1; then
              pip install --upgrade pip > /dev/null 2>&1
              pip install build
            fi
            if ! pip show opencv-python pyyaml pytest > /dev/null 2>&1; then
              echo "Installing test dependencies..."
              pip install opencv-python pyyaml pytest
            fi

            echo "dt-apriltags development environment"
            echo "Python: $(python --version)"
            echo "CMake: $(cmake --version | head -n1)"
            echo ""
            echo "Commands:"
            echo "  pip install -e .                              # Install in editable mode"
            echo "  pip install .                                 # Install package"
            echo "  python -m build                               # Build wheel"
            echo "  python test/test.py                           # Run tests"
          '';
        };
      }
    );
}
