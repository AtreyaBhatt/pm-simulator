{ pkgs, lib, config, inputs, ... }:

{
  packages = [ pkgs.git ];
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = builtins.readFile ./requirements.txt;
  };
}
