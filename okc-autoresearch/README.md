# OKC autoresearch control plane

Temporary branch for the Open Kernel Challenge campaign. It does not modify the Emmy project on main.

The workflow:
1. installs the official OKC CLI on a networked GitHub runner;
2. starts GitHub device login and posts the one-time code as an issue;
3. after approval, downloads the public H200 control lineage;
4. generates A06 (B=1 attention split sweep) and M06 (M=1 MoE route specialization);
5. runs unscored `okc check` only and uploads logs/profiles.

No `okc submit` is invoked in Wave 1.
