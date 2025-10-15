from __future__ import annotations


def main() -> None:
    import argparse
    import subprocess
    import sys
    from pathlib import Path
    import os

    parser = argparse.ArgumentParser(
        description="Run the lariat Snakemake pipeline (wrapper).",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Path to Snakemake config file (YAML/JSON)",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Run on a SLURM cluster",
    )
    parser.add_argument(
        "--partition",
        default="short,park",
        help="SLURM partition to use",
    )
    parser.add_argument(
        "--account",
        default="park",
        help="SLURM account to use",
    )
    parser.add_argument(
        "--cores",
        "-c",
        type=int,
        default=1,
        help="Number of CPU cores to use",
    )

    # Parse known args; anything else should be forwarded to Snakemake
    args, unknown = parser.parse_known_args()

    config_path = Path(args.config).expanduser()

    pipeline_dir = Path(__file__).resolve().parent
    snakefile = pipeline_dir / "main.smk"
    scripts_dir = pipeline_dir / "scripts"

    if not snakefile.exists():
        print(f"Error: Snakefile not found at: {snakefile}")
        sys.exit(2)

    if not config_path.exists():
        print(
            f"Warning: Config file not found at: {config_path} (Snakemake may still handle this)"
        )

    # Base snakemake command
    cmd = [
        "snakemake",
        "-s",
        str(snakefile),
        "--configfile",
        str(config_path.resolve()),
        "--use-conda",
        "--rerun-incomplete",
        "--cores",
        str(args.cores),
        "--directory",
        str(os.getcwd()),
    ]

    if args.cluster: 
        cmd.extend(
            [
                '--executor','slurm',
                '--default-resources',
                f'slurm_partition={args.partition}',
                f'slurm_account={args.account}',
                '--latency-wait','60',
                '--restart-times','3',
            ]
        )
    else:
        cmd.extend(
            [
                '--latency-wait', '5', 
                '--restart-times', '0',
            ]
        )

    # Forward all user-specified args
    cmd.extend(unknown)

    cmd.extend(["--config", f"scripts={scripts_dir}"])

    try:
        # Run within the pipeline directory so relative paths in the Snakefile resolve
        result = subprocess.run(cmd, cwd=str(pipeline_dir))
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(
            "Error: 'snakemake' executable not found. Please install Snakemake and ensure it is on your PATH."
        )
        sys.exit(127)


if __name__ == "__main__":
    main()
