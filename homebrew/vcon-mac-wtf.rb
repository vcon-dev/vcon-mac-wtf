# typed: false
# frozen_string_literal: true

class VconMacWtf < Formula
  include Language::Python::Virtualenv

  desc "MLX Whisper transcription server for Apple Silicon with WTF/vCon support"
  homepage "https://github.com/vcon-dev/vcon-mac-wtf"
  url "https://files.pythonhosted.org/packages/c7/09/a366547b5cbce81c8c16670f04acc05a51f82adc9bfc2a886aa5c0f5035a/vcon_mac_wtf-0.1.0.tar.gz"
  sha256 "bcc069e1409e346534defae7725a5679e1fd92254427628dfe726e663d683382"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_create(libexec, "python3.12")
    system libexec/"bin/pip", "install", "-v", "--ignore-installed", buildpath
    bin.install_symlink libexec/"bin/vcon-mac-wtf"
  end

  test do
    assert_path_exists bin/"vcon-mac-wtf"
    system opt_libexec/"bin/python", "-c", "import vcon_mac_wtf; print('ok')"
  end
end
