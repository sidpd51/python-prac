import pytest

from study.loader import load

ex = load(__file__, "ex01_requirements_parser")
R = ex.Requirement

REQUIREMENTS = """\
# LLM stack
openai==1.51.0
LangChain_Core >= 0.3, <0.4      # loose!
uvicorn[standard,  socks]==0.30.6
uvloop==0.20.0 ; sys_platform != "win32"
mypkg @ https://example.com/mypkg-1.0.tar.gz
-r dev-requirements.txt
--index-url https://pypi.org/simple

numpy==2.1.1 \\
    --hash=sha256:abc
tiktoken
pydantic==2.*
"""


@pytest.mark.topic("py.packaging.dependencies")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LangChain_Core", "langchain-core"),
        ("Sentence.Transformers", "sentence-transformers"),
        ("ruamel__yaml", "ruamel-yaml"),
        ("torch", "torch"),
    ],
)
def test_normalize_name(raw, expected):
    assert ex.normalize_name(raw) == expected


@pytest.mark.topic("py.packaging.requirements")
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("openai==1.51.0", R("openai", specifier="==1.51.0")),
        ("LangChain_Core >= 0.3, <0.4  # loose", R("langchain-core", specifier=">=0.3,<0.4")),
        ("uvicorn[standard,  Socks]==0.30.6", R("uvicorn", ("socks", "standard"), "==0.30.6")),
        ("httpx[http2,http2]", R("httpx", ("http2",))),
        (
            'uvloop==0.20.0 ; sys_platform != "win32"',
            R("uvloop", specifier="==0.20.0", marker='sys_platform != "win32"'),
        ),
        ("mypkg @ https://example.com/p.tar.gz", R("mypkg", url="https://example.com/p.tar.gz")),
        ("tiktoken", R("tiktoken")),
        ("requests~=2.32", R("requests", specifier="~=2.32")),
        ("numpy==2.1.1 --hash=sha256:abc", R("numpy", specifier="==2.1.1")),
    ],
)
def test_parse_requirement(line, expected):
    assert ex.parse_requirement(line) == expected


@pytest.mark.topic("py.packaging.requirements", "py.packaging.pip")
@pytest.mark.parametrize(
    "line",
    ["", "   ", "# just a comment", "-r base.txt", "--index-url https://x", "-e ."],
)
def test_lines_without_requirement(line):
    assert ex.parse_requirement(line) is None


@pytest.mark.topic("py.packaging.requirements")
def test_hash_in_url_is_not_a_comment():
    req = ex.parse_requirement("pkg @ https://example.com/p.zip#sha256=abc  # note")
    assert req.url == "https://example.com/p.zip#sha256=abc"


@pytest.mark.topic("py.packaging.requirements")
@pytest.mark.parametrize("line", ["==1.0", "pkg[extra==1.0", "pkg =1.0", "pkg>=", "pkg >= 1.0, foo"])
def test_malformed(line):
    with pytest.raises(ValueError):
        ex.parse_requirement(line)


@pytest.mark.topic("py.packaging.requirements", "py.packaging.pip")
def test_parse_requirements_file():
    reqs = ex.parse_requirements(REQUIREMENTS)
    assert [r.name for r in reqs] == [
        "openai",
        "langchain-core",
        "uvicorn",
        "uvloop",
        "mypkg",
        "numpy",
        "tiktoken",
        "pydantic",
    ]
    assert reqs[5] == R("numpy", specifier="==2.1.1"), "line continuation + --hash"


@pytest.mark.topic("py.packaging.dependencies")
def test_unpinned():
    reqs = ex.parse_requirements(REQUIREMENTS)
    assert ex.unpinned(reqs) == ["langchain-core", "tiktoken", "pydantic"]
    assert ex.unpinned([R("a", specifier="==1.0,!=1.0.1")]) == ["a"]
