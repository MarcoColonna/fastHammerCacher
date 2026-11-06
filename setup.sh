# Source this:  source setup.sh [/path/to/Hammer-install]
# Needs a Hammer build with python bindings, plus numpy.
# The Hammer install dir can also be given as HAMMER_DIR=/path/to/Hammer-install.
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_hdir="${1:-$HAMMER_DIR}"

if [ -n "$_hdir" ]; then
    if [ -f "$_hdir/bin/thishammer.sh" ]; then
        source "$_hdir/bin/thishammer.sh"
    else
        echo "setup.sh: $_hdir/bin/thishammer.sh not found" >&2
    fi
fi
export PYTHONPATH="$_here${PYTHONPATH:+:$PYTHONPATH}"

if python -c "import hammer, numpy" 2>/dev/null; then
    echo "OK: hammer from $(python -c 'import hammer, os; print(os.path.dirname(hammer.__file__))')"
else
    echo "setup.sh: cannot import hammer and/or numpy with $(command -v python)." >&2
    echo "  Activate a python env with numpy and pass the Hammer install dir:" >&2
    echo "  source setup.sh /path/to/Hammer-install" >&2
fi
unset _here _hdir
