import io, json, re, math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

st.set_page_config(page_title="Cyber Assurance Predictive Intelligence", layout="wide")

# -----------------------------
# Configuration
# -----------------------------
SEVERITY_WEIGHT = {"critical": 100, "high": 75, "medium": 45, "low": 20, "info": 5, "informational": 5}
STATUS_CLOSED = {"closed", "resolved", "done", "remediated"}
STATUS_ACCEPTED = {"risk accepted", "accepted", "exception", "waiver"}

CONTROL_TAXONOMY = {
    "Identity & Access Management": {
        "keywords": ["active directory", "ad privilege", "rbac", "access", "domain admin", "least privilege", "identity", "account", "user privilege"],
        "subcontrols": ["RBAC", "AD Privilege", "Least Privilege", "Account Governance"],
        "criticality": 1.25,
        "threat_class": "Identity Threat"
    },
    "Authentication Security": {
        "keywords": ["mfa", "multi-factor", "password", "authentication", "sso", "login", "credential", "vpn users"],
        "subcontrols": ["MFA", "Password Policy", "SSO", "Credential Protection"],
        "criticality": 1.30,
        "threat_class": "Credential / Initial Access Threat"
    },
    "Privileged Access Security": {
        "keywords": ["pam", "privileged", "admin account", "shared admin", "root", "sudo", "administrator"],
        "subcontrols": ["PAM", "Shared Admin Control", "Privileged Session", "Privilege Governance"],
        "criticality": 1.35,
        "threat_class": "Privilege Abuse Threat"
    },
    "Network Segmentation": {
        "keywords": ["firewall", "any-any", "segmentation", "vlan", "network access", "broad rule", "east-west", "flat network"],
        "subcontrols": ["Firewall Rules", "VLAN Segmentation", "Microsegmentation", "East-West Restriction"],
        "criticality": 1.25,
        "threat_class": "Lateral Movement Threat"
    },
    "East-West Traffic Security": {
        "keywords": ["kubernetes network polic", "network policy", "pod-to-pod", "pod communication", "service mesh", "namespace", "openshift network"],
        "subcontrols": ["Kubernetes NetworkPolicy", "Pod Isolation", "Service Mesh", "Namespace Segmentation"],
        "criticality": 1.20,
        "threat_class": "Container Lateral Movement Threat"
    },
    "Cryptographic Controls": {
        "keywords": ["tls", "mtls", "encryption", "cipher", "certificate", "unencrypted", "transport", "service-to-service encryption"],
        "subcontrols": ["TLS", "mTLS", "Certificate Control", "Encryption in Transit"],
        "criticality": 1.15,
        "threat_class": "Data Interception Threat"
    },
    "Endpoint Protection": {
        "keywords": ["edr", "av", "antivirus", "endpoint", "application control", "carbon black", "trellix", "agent"],
        "subcontrols": ["EDR", "AV", "Application Control", "Endpoint Hardening"],
        "criticality": 1.20,
        "threat_class": "Malware / Endpoint Threat"
    },
    "Detection & Monitoring": {
        "keywords": ["logging", "log retention", "siem", "monitoring", "alert", "detection", "audit log", "forensic"],
        "subcontrols": ["Logging", "SIEM", "Detection Coverage", "Retention"],
        "criticality": 1.10,
        "threat_class": "Detection Gap / Defense Evasion Threat"
    },
    "Security Hardening": {
        "keywords": ["cis", "hardening", "baseline", "secure configuration", "benchmark", "unused service", "default configuration"],
        "subcontrols": ["CIS Benchmark", "Secure Baseline", "Service Hardening", "Configuration Standard"],
        "criticality": 1.10,
        "threat_class": "Configuration Weakness Threat"
    },
    "Vulnerability Management": {
        "keywords": ["patch", "vulnerability", "cve", "kev", "outdated", "obsolete", "eol", "eos"],
        "subcontrols": ["Patch Management", "CVE Exposure", "KEV", "Lifecycle"],
        "criticality": 1.25,
        "threat_class": "Exploit / Vulnerability Threat"
    },
    "Cloud Security": {
        "keywords": ["cloud", "bucket", "s3", "storage account", "public storage", "azure", "aws", "gcp", "publicly accessible"],
        "subcontrols": ["Cloud Storage", "Cloud IAM", "Public Exposure", "Cloud Configuration"],
        "criticality": 1.25,
        "threat_class": "Cloud Exposure Threat"
    },
    "Container & Kubernetes Security": {
        "keywords": ["openshift", "kubernetes", "container", "image", "pod", "cluster", "runtime", "registry", "namespace"],
        "subcontrols": ["Container Runtime", "Image Security", "Cluster Security", "Namespace Control"],
        "criticality": 1.20,
        "threat_class": "Container Platform Threat"
    },
    "API Security": {
        "keywords": ["api", "token", "jwt", "oauth", "static token", "api authentication", "session", "rest", "soap"],
        "subcontrols": ["API Authentication", "Token Security", "Authorization", "Session Management"],
        "criticality": 1.25,
        "threat_class": "API Abuse Threat"
    },
    "Data Protection": {
        "keywords": ["dlp", "data leakage", "sensitive data", "classification", "pii", "confidential", "exfiltration"],
        "subcontrols": ["DLP", "Classification", "Data Leakage", "Sensitive Data Control"],
        "criticality": 1.20,
        "threat_class": "Data Exposure Threat"
    },
    "Email Security": {
        "keywords": ["email", "spf", "dkim", "dmarc", "phishing", "bec", "mail gateway"],
        "subcontrols": ["SPF", "DKIM", "DMARC", "Anti-Phishing", "BEC"],
        "criticality": 1.15,
        "threat_class": "Phishing / BEC Threat"
    },
    "Remote Access Security": {
        "keywords": ["vpn", "remote access", "external remote", "rdp", "citrix", "forticlient", "remote service"],
        "subcontrols": ["VPN", "External Remote Services", "Remote Admin", "Secure Access"],
        "criticality": 1.35,
        "threat_class": "External Access Threat"
    },
    "Third-Party Security": {
        "keywords": ["vendor", "third party", "outsourc", "partner", "integration", "supplier"],
        "subcontrols": ["Vendor Access", "Third-Party Integration", "Supplier Risk", "Contractual Security"],
        "criticality": 1.15,
        "threat_class": "Supply Chain Threat"
    },
    "DevSecOps": {
        "keywords": ["pipeline", "cicd", "gitlab", "argocd", "tekton", "deployment", "source code", "sast", "dast"],
        "subcontrols": ["CI/CD Security", "Pipeline Gate", "Code Security", "Deployment Control"],
        "criticality": 1.15,
        "threat_class": "Software Supply Chain Threat"
    },
    "Backup & Recovery": {
        "keywords": ["backup", "restore", "immutable", "recovery", "dr", "ransomware recovery"],
        "subcontrols": ["Backup", "Immutable Backup", "Restore Testing", "DR"],
        "criticality": 1.20,
        "threat_class": "Resilience / Ransomware Impact Threat"
    },
    "Asset Management": {
        "keywords": ["asset", "inventory", "cmdb", "unmanaged", "device42", "unknown server", "eol"],
        "subcontrols": ["Inventory", "CMDB", "Unmanaged Asset", "Lifecycle"],
        "criticality": 1.10,
        "threat_class": "Exposure Management Threat"
    },
    "Database Security": {
        "keywords": ["database", "oracle", "sql", "db", "privilege", "schema", "dba"],
        "subcontrols": ["DB Access", "DB Hardening", "DB Encryption", "DB Audit"],
        "criticality": 1.20,
        "threat_class": "Database Compromise Threat"
    },
    "Web Application Security": {
        "keywords": ["web", "owasp", "xss", "sql injection", "csrf", "application vulnerability", "session timeout"],
        "subcontrols": ["OWASP", "Input Validation", "Session Security", "Web Hardening"],
        "criticality": 1.20,
        "threat_class": "Web Application Threat"
    },
    "Security Governance": {
        "keywords": ["policy", "procedure", "governance", "approval", "risk acceptance", "ola", "sla", "process"],
        "subcontrols": ["Policy", "Procedure", "SLA/OLA", "Risk Governance"],
        "criticality": 1.00,
        "threat_class": "Governance Weakness"
    },
}

# ---------------------------------------------------------------------------
# MITRE ATT&CK control-to-technique mapping.
#
# Design principles:
#
#  1. Techniques represent what an ADVERSARY does. A finding describes a
#     CONTROL GAP; the mapped technique is the adversary action that gap
#     most directly enables.
#
#  2. Each entry: (technique_id, name, tactic, trigger_keywords, reason, base_confidence)
#
#  3. Trigger keywords are the primary quality gate. They must be SPECIFIC
#     enough that only findings genuinely relevant to the technique will match.
#     Generic words that appear across all findings in a domain (e.g. "access",
#     "user", "network") are excluded — those produce false positives at scale.
#     The deny list is NOT a substitute for weak triggers.
#
#  4. Confidence is deliberately conservative. Weak signal → low confidence
#     → falls below the 0.68 threshold → "No reliable mapping". That is the
#     correct and honest outcome for ambiguous findings.
#
#  5. Sub-technique differentiation: where two domains naturally share a
#     parent technique (e.g. T1195 for both DevSecOps and Third-Party),
#     the correct sub-technique is used to differentiate them at the
#     mapping level rather than relying on deny rules.
# ---------------------------------------------------------------------------

CONTROL_ATTACK_MAP = {
    "Authentication Security": [
        # Triggers: password-policy-specific tokens only. "password" alone
        # is too generic — must co-occur with policy/control context.
        ("T1110.003", "Password Spraying", "Credential Access",
         ["password policy", "account lockout", "weak password", "password complexity",
          "brute force", "credential stuffing"],
         "Absent or weak password controls directly enable password spraying against valid accounts.", 0.90),
        # Triggers: MFA + remote-service specific. "mfa" alone on an internal
        # finding does not imply external remote service exposure.
        ("T1133", "External Remote Services", "Initial Access",
         ["mfa enforcement", "vpn mfa", "remote access mfa", "remote login mfa",
          "multi-factor remote", "mfa bypass"],
         "Remote access without MFA enforcement is the primary initial-access vector via exposed remote services.", 0.88),
        # Triggers: explicit credential-abuse tokens. "credential" is kept but
        # requires co-occurrence with abuse/bypass/theft context.
        ("T1078", "Valid Accounts", "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
         ["credential theft", "authentication bypass", "sso bypass", "stolen credential",
          "credential reuse", "account takeover"],
         "Weak authentication controls increase the likelihood of adversaries abusing valid credentials.", 0.72),
    ],
    "Remote Access Security": [
        # Triggers: VPN/remote-service product names + weak-control terms.
        # "remote" alone is too broad across a full risk register.
        ("T1133", "External Remote Services", "Initial Access",
         ["vpn", "forticlient", "citrix", "anyconnect", "remote access gateway",
          "ssl vpn", "ipsec vpn", "remote access portal"],
         "Exposed remote access services with weak authentication are a primary initial-access vector.", 0.92),
        # Triggers: protocol-specific. "remote admin" kept; generic "remote" removed.
        ("T1021", "Remote Services", "Lateral Movement",
         ["rdp", "ssh", "winrm", "remote desktop", "remote administration",
          "psremoting", "smb lateral"],
         "Insecure remote administration protocols can be abused for lateral movement once initial access is established.", 0.80),
    ],
    "Privileged Access Security": [
        # Triggers: explicit privileged-account-naming tokens. "privileged" alone
        # removed — it appears in nearly every security finding.
        ("T1078.002", "Domain Accounts", "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
         ["shared admin", "domain admin", "local admin", "privileged account",
          "admin account sharing", "domain administrator"],
         "Weak privileged account governance enables adversaries to abuse domain or local admin accounts.", 0.88),
        # Triggers: token/impersonation specific — not triggered by generic "privilege".
        ("T1134", "Access Token Manipulation", "Defense Evasion / Privilege Escalation",
         ["token impersonation", "runas", "sudo abuse", "privilege escalation token",
          "impersonation attack", "seimpersonateprivilege"],
         "Missing PAM controls or token governance may allow adversaries to manipulate access tokens for privilege escalation.", 0.76),
    ],
    "Identity & Access Management": [
        # Triggers: AD-specific terms only. "access" and "identity" removed —
        # they appear universally and add no discriminative value.
        ("T1078.002", "Domain Accounts", "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
         ["active directory", "domain admin", "ad privilege", "domain account",
          "group policy abuse", "ad group", "ad misconfiguration"],
         "Excessive Active Directory privileges increase the risk of domain account abuse.", 0.88),
        # Triggers: explicit over-permission tokens. "access rights" alone removed;
        # requires context indicating excessive grant, not just any access finding.
        ("T1078", "Valid Accounts", "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
         ["over-permissioned", "excessive privilege", "least privilege violation",
          "unnecessary access", "orphaned account", "excessive role"],
         "Excessive permissions enable adversaries to operate using valid credentials without raising alerts.", 0.78),
    ],
    "Network Segmentation": [
        # Triggers: specific segmentation-failure tokens. "firewall" alone removed —
        # it appears in many unrelated findings (e.g. firewall product names).
        ("T1021", "Remote Services", "Lateral Movement",
         ["any-any rule", "flat network", "missing segmentation", "unrestricted lateral",
          "broad firewall rule", "no vlan isolation", "east-west unrestricted"],
         "Weak internal segmentation allows adversaries to reach internal services for lateral movement.", 0.82),
        # Triggers: discovery-specific. "internal access" removed — too generic.
        ("T1046", "Network Service Discovery", "Discovery",
         ["flat network scan", "internal host discovery", "unrestricted internal reach",
          "open internal ports", "no microsegmentation"],
         "Flat networks enable adversaries to enumerate reachable internal services.", 0.74),
    ],
    "East-West Traffic Security": [
        # Triggers: container-isolation-specific. "container" alone removed —
        # it appears in Container & Kubernetes Security domain too.
        ("T1611", "Escape to Host", "Privilege Escalation",
         ["privileged pod", "host path mount", "host pid", "host network",
          "docker socket mount", "no seccomp", "root container"],
         "Overly permissive pod configurations or host mounts increase the risk of container-to-host breakout.", 0.80),
        # Triggers: network-policy-specific terms for lateral movement context.
        ("T1210", "Exploitation of Remote Services", "Lateral Movement",
         ["pod-to-pod unrestricted", "missing network policy", "namespace isolation gap",
          "no networkpolicy", "open east-west", "service mesh bypass"],
         "Missing Kubernetes NetworkPolicies allow unrestricted pod-to-pod traffic, enabling lateral movement.", 0.74),
    ],
    "Cryptographic Controls": [
        # Triggers: unencrypted-traffic-specific. "tls" kept as it directly signals
        # the transport layer concern. "encryption" alone removed — too broad.
        ("T1040", "Network Sniffing", "Credential Access / Discovery",
         ["unencrypted traffic", "cleartext", "plaintext protocol", "no tls",
          "mtls missing", "service-to-service cleartext", "http not https"],
         "Unencrypted traffic can be intercepted by an attacker with network visibility.", 0.92),
        # Triggers: protocol-downgrade / weak-cipher specific.
        ("T1557", "Adversary-in-the-Middle", "Credential Access / Collection",
         ["tls 1.0", "tls 1.1", "weak cipher", "ssl downgrade", "certificate validation",
          "self-signed cert", "no certificate pinning", "mitm"],
         "Weak transport configuration or missing certificate validation enables adversary-in-the-middle attacks.", 0.86),
    ],
    "Detection & Monitoring": [
        # Triggers: monitoring-gap-specific. "detection" alone removed — generic.
        # T1562.006 (Indicator Blocking) is what adversaries exploit when monitoring is absent.
        ("T1562.006", "Indicator Blocking", "Defense Evasion",
         ["log retention gap", "siem gap", "missing siem", "no centralised logging",
          "monitoring blind spot", "alert suppression", "detection gap"],
         "Monitoring and logging gaps allow adversaries to operate undetected — exploited via indicator blocking.", 0.82),
        # Triggers: forensic/audit-evidence-specific only.
        ("T1070", "Indicator Removal", "Defense Evasion",
         ["audit log", "forensic evidence", "log deletion", "log tampering",
          "event log cleared", "no log integrity", "log forwarding gap"],
         "Weak audit log controls reduce the ability to detect or reconstruct attacker activity.", 0.72),
    ],
    "Security Hardening": [
        # Triggers: internet-exposure + hardening-failure context. "public" alone removed.
        ("T1190", "Exploit Public-Facing Application", "Initial Access",
         ["internet-facing service", "exposed management interface", "public-facing",
          "default credentials", "hardening gap", "cis benchmark gap", "secure baseline missing"],
         "Missing hardening on internet-facing services increases susceptibility to exploitation.", 0.82),
        # Triggers: unused/legacy service specific — explicit execution-path tokens.
        ("T1059", "Command and Scripting Interpreter", "Execution",
         ["unused service enabled", "legacy protocol enabled", "unnecessary component",
          "default scripting enabled", "unrestricted script execution"],
         "Unused or default services left enabled may provide adversaries with script execution paths.", 0.68),
    ],
    "Vulnerability Management": [
        # Triggers: CVE/patch-specific. "vulnerability" kept but paired with
        # exposure-context tokens for specificity.
        ("T1190", "Exploit Public-Facing Application", "Initial Access",
         ["unpatched cve", "known cve", "public-facing vulnerability", "internet-exposed cve",
          "exploit available", "kev", "zero day", "patch missing"],
         "Known unpatched vulnerabilities on exposed services are a primary exploitation vector.", 0.88),
        # Triggers: local escalation specific — not triggered by general patch findings.
        ("T1068", "Exploitation for Privilege Escalation", "Privilege Escalation",
         ["local privilege escalation", "local vulnerability", "kernel vulnerability",
          "eop cve", "elevation of privilege cve", "local exploit"],
         "Unpatched local privilege-escalation vulnerabilities can be abused after initial access.", 0.78),
    ],
    "Cloud Security": [
        # Triggers: cloud-storage-misconfiguration specific.
        ("T1530", "Data from Cloud Storage", "Collection",
         ["public bucket", "public blob", "s3 public", "storage account public",
          "publicly accessible storage", "misconfigured cloud storage", "open bucket"],
         "Misconfigured public cloud storage allows adversaries to collect sensitive data directly.", 0.95),
        # Triggers: cloud-identity-specific. "azure" alone removed — appears in
        # many non-IAM findings (e.g. Azure Kubernetes Service, Azure networking).
        ("T1078.004", "Cloud Accounts", "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
         ["cloud iam", "aws iam", "azure ad role", "service principal", "managed identity",
          "cloud account abuse", "overprivileged cloud role"],
         "Weak cloud identity controls may allow adversaries to abuse cloud accounts.", 0.84),
    ],
    "Container & Kubernetes Security": [
        # Triggers: container-runtime-weakness specific. "container" alone removed —
        # too generic; must be paired with weakness-specific token.
        ("T1611", "Escape to Host", "Privilege Escalation",
         ["privileged container", "host path mount", "docker socket", "host pid namespace",
          "root in container", "no seccomp profile", "container escape"],
         "Privileged containers or unsafe host mounts increase the risk of container-to-host breakout.", 0.82),
        # Triggers: image/registry-weakness specific.
        ("T1610", "Deploy Container", "Defense Evasion / Execution",
         ["unscanned image", "unsigned image", "malicious base image", "untrusted registry",
          "image integrity gap", "no image signing", "registry misconfiguration"],
         "Weak container image governance may allow deployment of attacker-controlled images.", 0.76),
    ],
    "API Security": [
        # Triggers: token/key-governance specific. "api" alone removed — appears
        # universally; must co-occur with token/auth-weakness terms.
        ("T1528", "Steal Application Access Token", "Credential Access",
         ["static api key", "long-lived token", "jwt without expiry", "api key rotation",
          "hardcoded api key", "bearer token exposure", "oauth misconfiguration"],
         "Long-lived or unrotated API tokens increase the risk of token theft and replay attacks.", 0.92),
        # Triggers: unauthenticated-endpoint specific.
        ("T1190", "Exploit Public-Facing Application", "Initial Access",
         ["unauthenticated api", "unprotected endpoint", "public api", "api without auth",
          "exposed api gateway", "api injection"],
         "Unauthenticated or exposed API endpoints are exploitable for initial access.", 0.74),
    ],
    "Data Protection": [
        # Triggers: data-loss-specific. "sensitive data" kept but requires
        # exfiltration/leakage context, not just classification mentions.
        ("T1048", "Exfiltration Over Alternative Protocol", "Exfiltration",
         ["data exfiltration", "dlp gap", "unmonitored exfil", "data leakage channel",
          "pii exfiltration", "uncontrolled data transfer", "missing dlp"],
         "Weak DLP controls increase the risk of data exfiltration over unmonitored protocols.", 0.80),
        # Triggers: cloud/SaaS upload specific — not triggered by generic data findings.
        ("T1567", "Exfiltration Over Web Service", "Exfiltration",
         ["personal cloud upload", "unauthorized saas", "onedrive exfil",
          "dropbox upload", "sharepoint oversharing", "cloud storage exfil"],
         "Without DLP controls, adversaries may exfiltrate data via legitimate web services.", 0.72),
    ],
    "Email Security": [
        # Triggers: email-threat specific. "email" alone removed — too generic.
        ("T1566", "Phishing", "Initial Access",
         ["phishing campaign", "bec", "email attachment", "malicious link email",
          "spear phishing", "email lure", "no anti-phishing"],
         "Weak email controls increase susceptibility to phishing and business email compromise.", 0.90),
        # Triggers: email-auth-record specific.
        ("T1566.002", "Spearphishing Link", "Initial Access",
         ["dmarc missing", "spf missing", "dkim missing", "domain spoofing",
          "email spoofing", "sender policy", "email authentication gap"],
         "Missing email authentication records enable domain spoofing in spearphishing campaigns.", 0.82),
    ],
    "Endpoint Protection": [
        # Triggers: EDR/AV-product specific. "agent" alone removed — generic.
        ("T1562.001", "Disable or Modify Tools", "Defense Evasion",
         ["edr gap", "av gap", "missing edr", "no antivirus", "endpoint protection missing",
          "trellix", "carbon black", "sentinelone", "crowdstrike gap"],
         "Endpoint protection gaps allow adversaries to disable or bypass security tooling.", 0.84),
        # Triggers: script-execution specific — not triggered by generic "endpoint".
        ("T1059", "Command and Scripting Interpreter", "Execution",
         ["powershell unrestricted", "script execution policy", "no application control",
          "applocker gap", "wdac gap", "unrestricted script", "bash unrestricted"],
         "Without endpoint controls, adversaries execute malicious scripts via interpreters undetected.", 0.74),
    ],
    "Backup & Recovery": [
        # Triggers: ransomware-backup specific.
        ("T1486", "Data Encrypted for Impact", "Impact",
         ["backup not immutable", "no immutable backup", "ransomware recovery gap",
          "backup encryption gap", "unprotected backup", "bcdr gap", "backup accessible"],
         "Weak backup controls increase the ransomware encryption impact on business continuity.", 0.84),
        # Triggers: recovery-inhibition specific.
        ("T1490", "Inhibit System Recovery", "Impact",
         ["shadow copy unprotected", "volume shadow gap", "no backup integrity",
          "recovery point gap", "dr plan untested", "backup deletion risk"],
         "Adversaries delete or corrupt backups to prevent recovery — weak controls amplify this impact.", 0.76),
    ],
    "DevSecOps": [
        # Triggers: CI/CD-pipeline specific. "deployment" alone removed — generic.
        ("T1195.002", "Compromise Software Supply Chain", "Initial Access",
         ["cicd pipeline", "gitlab pipeline", "pipeline integrity", "argocd",
          "tekton", "build artifact", "pipeline gate missing", "sast missing"],
         "Weak CI/CD pipeline controls allow adversaries to inject malicious code during build or deployment.", 0.84),
        # Triggers: deployment-tool specific.
        ("T1072", "Software Deployment Tools", "Execution / Lateral Movement",
         ["ansible misconfiguration", "helm misconfiguration", "puppet gap",
          "chef gap", "deployment tool access", "iac misconfiguration"],
         "Poorly governed deployment tooling can distribute malicious payloads across environments.", 0.72),
    ],
    "Third-Party Security": [
        # Triggers: dependency/supplier specific. "vendor" alone removed — too
        # generic; must co-occur with dependency/supply-chain context tokens.
        ("T1195.001", "Compromise Software Dependencies and Development Tools", "Initial Access",
         ["third-party dependency", "unverified library", "open source risk",
          "supplier software", "dependency confusion", "package integrity", "npm risk"],
         "Unverified third-party dependencies increase the risk of software supply chain compromise.", 0.80),
        # Triggers: vendor remote-access specific.
        ("T1133", "External Remote Services", "Initial Access",
         ["vendor remote access", "third-party vpn", "partner access uncontrolled",
          "supplier remote", "vendor session unmonitored", "third-party rdp"],
         "Uncontrolled vendor remote access can be exploited as an initial access vector.", 0.76),
    ],
    "Asset Management": [
        # Triggers: untracked-asset specific. "asset" and "inventory" alone removed.
        ("T1592", "Gather Victim Host Information", "Reconnaissance",
         ["unmanaged asset", "shadow it", "unknown device", "cmdb gap",
          "undiscovered host", "rogue device", "asset inventory gap"],
         "Untracked assets provide adversaries with undetected attack surface.", 0.74),
        # Triggers: EOL/unmanaged-specific for exploitation context.
        ("T1190", "Exploit Public-Facing Application", "Initial Access",
         ["eol system", "end-of-life software", "unsupported os", "unmanaged server",
          "undiscovered internet-facing", "unpatched unmanaged"],
         "Unmanaged or EOL systems are prime targets for exploitation.", 0.70),
    ],
    "Database Security": [
        # Triggers: DB-account-specific. "database" alone removed.
        ("T1078", "Valid Accounts", "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
         ["dba account", "sa account", "database privilege", "schema owner",
          "excessive db role", "db account misconfiguration", "oracle dba"],
         "Excessive database account privileges enable adversaries to abuse valid DB credentials.", 0.82),
        # Triggers: DB-data-collection specific.
        ("T1005", "Data from Local System", "Collection",
         ["unencrypted database", "database plaintext", "sensitive db table",
          "cardholder data db", "pii in database", "financial data unprotected"],
         "Weak database access controls allow adversaries to query and collect sensitive data.", 0.78),
    ],
    "Web Application Security": [
        # Triggers: OWASP-specific vulnerability tokens.
        ("T1190", "Exploit Public-Facing Application", "Initial Access",
         ["sql injection", "xss vulnerability", "csrf", "owasp finding",
          "injection vulnerability", "application vulnerability", "broken access control"],
         "Web application vulnerabilities enable direct exploitation of internet-facing services.", 0.88),
        # Triggers: session-specific — not triggered by generic "web" findings.
        ("T1185", "Browser Session Hijacking", "Collection",
         ["session fixation", "session token reuse", "cookie misconfiguration",
          "missing secure flag", "session timeout missing", "insecure session management"],
         "Weak session management allows adversaries to hijack authenticated user sessions.", 0.74),
    ],
}

# ---------------------------------------------------------------------------
# Deny rules — MINIMAL by design.
#
# Philosophy:
#   Tight trigger keywords are the PRIMARY quality gate. The deny list is a
#   last-resort mechanism used ONLY when two domains share near-identical
#   vocabulary and keyword specificity alone cannot distinguish them — i.e.
#   sub-technique disambiguation.
#
#   DO NOT add a deny pre-emptively. Every deny is a potential suppressed
#   true positive on a future dataset. If a finding legitimately involves a
#   technique, the confidence threshold (0.68) and trigger specificity should
#   handle it — not a hard block.
#
# Current justified denies (3 only):
#
#  Data Protection → T1041 (Exfiltration Over C2 Channel)
#    T1041 presupposes an attacker has already established C2 infrastructure.
#    A DLP/classification gap finding cannot imply that. T1048 and T1567
#    cover the concern correctly without presupposing C2. Safe to deny because
#    no Data Protection finding would legitimately map to C2 exfil specifically.
#
#  DevSecOps → T1195.001 (Compromise Software Dependencies)
#    CI/CD pipeline findings ("pipeline", "artifact", "build", "gitlab") share
#    vocabulary with dependency/supply-chain findings. Without this deny,
#    T1195.001 can fire on pipeline findings that should map to T1195.002.
#    Keyword specificity alone is insufficient because both sub-techniques
#    legitimately reference "package", "artifact", and "build" contexts.
#
#  Third-Party Security → T1195.002 (Compromise Software Supply Chain)
#    Mirror of above. Vendor/supplier findings share vocabulary with pipeline
#    findings. Hard deny ensures T1195.002 does not fire in the Third-Party
#    domain where T1195.001 is the structurally correct mapping.
#
# All other domains have NO deny list. Cross-domain technique overlap is
# handled entirely by trigger keyword specificity and the 0.68 confidence
# floor — this maximises recall on diverse real-world datasets.
# ---------------------------------------------------------------------------
DENY_TECHNIQUES_BY_CONTROL = {
    "Data Protection": {
        "T1041",     # C2 exfil presupposes C2 infrastructure — T1048/T1567 are correct
    },
    "DevSecOps": {
        "T1195.001", # Sub-technique disambiguation: dependency poisoning → Third-Party domain
    },
    "Third-Party Security": {
        "T1195.002", # Sub-technique disambiguation: pipeline compromise → DevSecOps domain
    },
}


def normalize_cols(df):
    df = df.copy()
    aliases = {
        "finding id": "Finding ID", "id": "Finding ID", "risk id": "Finding ID",
        "finding": "Title", "finding title": "Title", "title": "Title", "name": "Title",
        "description": "Description", "details": "Description", "risk description": "Description",
        "severity": "Severity", "risk severity": "Severity", "rating": "Severity",
        "status": "Status", "state": "Status",
        "owner": "Owner", "assignee": "Owner", "responsible team": "Owner",
        "business unit": "Business Unit", "bu": "Business Unit", "department": "Business Unit",
        "domain": "Domain", "technology": "Technology", "root cause": "Root Cause", "risk category": "Risk Category",
        "created date": "Created Date", "creation date": "Created Date", "identified date": "Created Date",
        "due date": "Due Date", "target date": "Due Date", "remediation date": "Due Date",
        "closure date": "Closure Date", "closed date": "Closure Date",
        "environment": "Environment", "asset criticality": "Asset Criticality", "internet facing": "Internet Facing", "exposure": "Exposure"
    }
    new_cols = {}
    for c in df.columns:
        key = str(c).strip().lower()
        new_cols[c] = aliases.get(key, str(c).strip())
    df = df.rename(columns=new_cols)
    for col in ["Finding ID","Title","Description","Severity","Status","Owner","Business Unit","Domain","Technology","Root Cause","Risk Category","Created Date","Due Date","Closure Date","Environment","Asset Criticality","Internet Facing","Exposure"]:
        if col not in df.columns:
            df[col] = ""
    if df["Finding ID"].replace("", np.nan).isna().all():
        df["Finding ID"] = [f"F-{i+1:04d}" for i in range(len(df))]
    return df


def load_tabular(upload):
    if upload is None:
        return None
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def load_json_file(upload):
    """Load JSON from a Streamlit upload object or local path."""
    if upload is None:
        return None
    if hasattr(upload, "read"):
        raw = upload.read()
        try:
            upload.seek(0)
        except Exception:
            pass
        return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    with open(upload, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_mitre_stix(stix_json):
    """Extract enterprise ATT&CK technique metadata from official MITRE STIX JSON.

    The mapper remains conservative and rule-constrained. Uploaded STIX is used to
    validate technique IDs and replace hardcoded names/tactics with authoritative
    metadata from the MITRE dataset.
    """
    if not stix_json:
        return {}, pd.DataFrame()
    objects = stix_json.get("objects", []) if isinstance(stix_json, dict) else []
    phase_to_tactic = {}
    for obj in objects:
        if obj.get("type") == "x-mitre-tactic":
            shortname = obj.get("x_mitre_shortname", "")
            phase_to_tactic[shortname] = obj.get("name", shortname).title()
    rows = []
    lookup = {}
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        ext_refs = obj.get("external_references", []) or []
        attack_id = ""
        url = ""
        for ref in ext_refs:
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
                attack_id = ref.get("external_id")
                url = ref.get("url", "")
                break
        if not attack_id:
            continue
        tactics = []
        for phase in obj.get("kill_chain_phases", []) or []:
            if phase.get("kill_chain_name") == "mitre-attack":
                pname = phase.get("phase_name", "")
                tactics.append(phase_to_tactic.get(pname, pname.replace("-", " ").title()))
        row = {
            "mitre_id": attack_id,
            "mitre_name": obj.get("name", ""),
            "mitre_tactics": ", ".join(dict.fromkeys([t for t in tactics if t])),
            "mitre_platforms": ", ".join(obj.get("x_mitre_platforms", []) or []),
            "mitre_detection": obj.get("x_mitre_detection", ""),
            "mitre_url": url,
            "mitre_description": obj.get("description", ""),
        }
        rows.append(row)
        lookup[attack_id] = row
    return lookup, pd.DataFrame(rows)


def apply_mitre_metadata(df, mitre_lookup=None):
    """Apply authoritative MITRE metadata when an uploaded ATT&CK STIX file is present.

    FIX: Original used df.loc[row.name, col] = value inside iterrows() — modifying
    a DataFrame while iterating it is unreliable and can silently drop updates on
    larger datasets. Replaced with list-accumulation pattern: collect all values
    into lists, then assign once after the loop.
    """
    df = df.copy()
    if not mitre_lookup:
        df["mitre_metadata_source"] = "Curated local mapping"
        df["mitre_url"] = ""
        df["mitre_platforms"] = ""
        df["mitre_detection"] = ""
        return df

    names, tactics, sources, urls, platforms, detections = [], [], [], [], [], []
    for _, row in df.iterrows():
        tid = str(row.get("mitre_primary_id", ""))
        meta = mitre_lookup.get(tid) if tid != "No reliable mapping" else None
        if meta:
            names.append(meta.get("mitre_name", row.get("mitre_primary_name", "")))
            tactics.append(meta.get("mitre_tactics") or row.get("mitre_tactic", ""))
            sources.append("Uploaded MITRE ATT&CK STIX")
            urls.append(meta.get("mitre_url", ""))
            platforms.append(meta.get("mitre_platforms", ""))
            detections.append(meta.get("mitre_detection", ""))
        else:
            names.append(row.get("mitre_primary_name", ""))
            tactics.append(row.get("mitre_tactic", ""))
            sources.append("Curated local mapping / Not mapped")
            urls.append("")
            platforms.append("")
            detections.append("")

    df["mitre_primary_name"] = names
    df["mitre_tactic"] = tactics
    df["mitre_metadata_source"] = sources
    df["mitre_url"] = urls
    df["mitre_platforms"] = platforms
    df["mitre_detection"] = detections
    return df


def text_of(row):
    parts = [row.get(c, "") for c in ["Title","Description","Domain","Technology","Root Cause","Risk Category","Environment","Exposure"]]
    return " ".join([str(x) for x in parts if pd.notna(x)]).lower()


def classify_control(row):
    txt = text_of(row)
    scores = []
    for domain, cfg in CONTROL_TAXONOMY.items():
        score = sum(1 for k in cfg["keywords"] if k in txt)
        # Boost exact sub-control patterns
        if domain == "Remote Access Security" and ("vpn" in txt or "remote access" in txt): score += 3
        if domain == "Cryptographic Controls" and ("encryption" in txt or "tls" in txt or "mtls" in txt): score += 3
        if domain == "Detection & Monitoring" and ("logging" in txt or "retention" in txt or "siem" in txt): score += 3
        if domain == "East-West Traffic Security" and ("network polic" in txt or "pod" in txt): score += 3
        if domain == "Security Hardening" and ("cis" in txt or "hardening" in txt): score += 3
        scores.append((domain, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    best, score = scores[0]
    if score <= 0:
        return "Security Governance", "General Governance", 0.35
    cfg = CONTROL_TAXONOMY[best]
    sub = cfg["subcontrols"][0]
    for s in cfg["subcontrols"]:
        if s.lower().split()[0] in txt or s.lower() in txt:
            sub = s
            break
    conf = min(0.95, 0.45 + score * 0.10)
    return best, sub, conf


def map_mitre(row):
    control = row.get("control_domain", "")
    txt = text_of(row)
    candidates = []
    deny = DENY_TECHNIQUES_BY_CONTROL.get(control, set())
    for tid, name, tactic, triggers, reason, base in CONTROL_ATTACK_MAP.get(control, []):
        if tid in deny:
            continue
        hits = [t for t in triggers if t in txt]
        if hits:
            confidence = min(0.97, base + 0.02*len(hits))
            candidates.append((tid, name, tactic, confidence, reason, ", ".join(hits)))
        elif base >= 0.84 and control in ["Cloud Security", "API Security"]:
            candidates.append((tid, name, tactic, base-0.12, reason, "control-domain context"))
    candidates.sort(key=lambda x: x[3], reverse=True)
    if not candidates or candidates[0][3] < 0.68:
        return pd.Series({
            "mitre_primary_id":"No reliable mapping", "mitre_primary_name":"No reliable mapping", "mitre_tactic":"Not mapped",
            "mitre_confidence":0.0, "mitre_confidence_label":"No reliable mapping",
            "mitre_reason":"ATT&CK mapping was not forced because the finding does not provide enough adversary-behavior context.",
            "mitre_matching_terms":"", "mitre_secondary":""
        })
    primary = candidates[0]
    secondary = ""
    if len(candidates) > 1 and candidates[1][3] >= 0.72:
        secondary = f"{candidates[1][0]} - {candidates[1][1]} ({candidates[1][3]:.0%})"
    label = "High" if primary[3] >= 0.85 else "Medium" if primary[3] >= 0.72 else "Low"
    return pd.Series({
        "mitre_primary_id": primary[0], "mitre_primary_name": primary[1], "mitre_tactic": primary[2],
        "mitre_confidence": round(primary[3], 3), "mitre_confidence_label": label,
        "mitre_reason": primary[4], "mitre_matching_terms": primary[5], "mitre_secondary": secondary
    })


def parse_dates(df):
    df = df.copy()
    for c in ["Created Date", "Due Date", "Closure Date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def is_closed(status):
    return str(status).strip().lower() in STATUS_CLOSED


def enrich_scores(df, cti=None, mitre_lookup=None):
    df = df.copy()
    today = pd.Timestamp.today().normalize()
    df = parse_dates(df)
    df["severity_points"] = (
        df["Severity"].astype(str).str.strip().str.lower()
        .map(SEVERITY_WEIGHT)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(35)
    )
    df["is_closed"] = df["Status"].apply(is_closed)
    df["is_risk_accepted"] = df["Status"].astype(str).str.lower().isin(STATUS_ACCEPTED)
    df["age_days"] = (today - df["Created Date"]).dt.days.fillna(0).clip(lower=0)
    df["days_overdue"] = (today - df["Due Date"]).dt.days.fillna(0)
    df.loc[df["days_overdue"] < 0, "days_overdue"] = 0
    df.loc[df["is_closed"], "days_overdue"] = 0
    df["sla_breached"] = (df["days_overdue"] > 0) & (~df["is_closed"])

    controls = df.apply(classify_control, axis=1, result_type="expand")
    df["control_domain"] = controls[0]
    df["sub_control"] = controls[1]
    df["control_classification_confidence"] = controls[2]
    df["control_criticality_multiplier"] = df["control_domain"].map(lambda d: CONTROL_TAXONOMY.get(d, {}).get("criticality", 1.0))
    df["threat_classification"] = df["control_domain"].map(lambda d: CONTROL_TAXONOMY.get(d, {}).get("threat_class", "General Cyber Risk"))

    mitre = df.apply(map_mitre, axis=1)
    df = pd.concat([df, mitre], axis=1)
    df = apply_mitre_metadata(df, mitre_lookup)

    # CTI mapping
    cti_map = map_cti(df, cti) if cti is not None and len(cti) else pd.DataFrame(index=df.index)
    for col in cti_map.columns:
        df[col] = cti_map[col]
    if "cti_match_score" not in df.columns:
        df["cti_match_score"] = 0.0
        df["matched_incident"] = "No CTI incident match"
        df["cti_business_meaning"] = "No relevant historical incident was found in the uploaded CTI dataset."
        df["cti_source"] = ""
        df["cti_match_terms"] = ""

    exposure_mult = df.apply(exposure_multiplier, axis=1)
    df["exposure_multiplier"] = exposure_mult
    # Guard: ensure all multiplier columns are numeric before formula to prevent silent NaN propagation
    for _col in ["severity_points", "control_criticality_multiplier", "exposure_multiplier"]:
        df[_col] = pd.to_numeric(df[_col], errors="coerce").fillna(1.0)
    cti_mult = 1 + (df["cti_match_score"].fillna(0).astype(float) * 0.30)
    overdue_mult = 1 + np.minimum(df["days_overdue"].fillna(0)/90, 0.40)
    age_mult = 1 + np.minimum(df["age_days"].fillna(0)/365, 0.25)
    accepted_discount = np.where(df["is_risk_accepted"], 0.90, 1.0)
    closed_discount = np.where(df["is_closed"], 0.35, 1.0)
    raw_score = df["severity_points"] * df["control_criticality_multiplier"] * exposure_mult * cti_mult * overdue_mult * age_mult * accepted_discount * closed_discount
    df["risk_score"] = np.clip(raw_score, 0, 100).round(1)
    df["breach_probability"] = (1 / (1 + np.exp(-((df["risk_score"]-55)/12))) * 100).round(1)
    df["risk_gauge"] = pd.cut(df["risk_score"], bins=[-1,30,60,80,101], labels=["Stable", "Elevated", "High Exposure", "Critical Exposure"])
    df["management_action"] = df.apply(recommended_action, axis=1)
    df["where_we_stand"] = df.apply(where_we_stand_row, axis=1)
    df["future_outlook"] = df.apply(future_outlook_row, axis=1)
    return df


def exposure_multiplier(row):
    txt = text_of(row)
    m = 1.0
    if any(x in txt for x in ["internet", "public", "external", "public-facing"]): m += 0.45
    if "vpn" in txt or "remote" in txt: m += 0.35
    if "production" in txt or "prod" in txt: m += 0.25
    if any(x in txt for x in ["critical", "payment", "core", "privileged"]): m += 0.20
    return min(m, 2.0)


def recommended_action(row):
    g = str(row.get("risk_gauge", ""))
    if g == "Critical Exposure":
        return "Immediate management attention: assign accountable owner, confirm compensating controls, and track remediation weekly."
    if g == "High Exposure":
        return "Prioritize remediation in the current cycle and validate whether temporary controls are operating effectively."
    if g == "Elevated":
        return "Track through normal remediation governance and prevent recurrence through baseline/control updates."
    return "Maintain monitoring and close through standard governance."


def where_we_stand_row(row):
    return f"{row['risk_gauge']} posture driven by {row['control_domain']} weakness; breach probability estimated at {row['breach_probability']}%."


def future_outlook_row(row):
    if row["risk_score"] >= 80:
        return "If not remediated, this may become a material exposure and contribute to incident likelihood within 30-90 days."
    if row["risk_score"] >= 60:
        return "If delayed, this may shift to critical exposure as age, SLA breach, or threat activity increases."
    return "Expected to remain manageable if remediation stays within SLA and no matching CTI pressure increases."


def _scale_cti_score(raw):
    """Non-linear CTI score scaling.

    FIX: Original used a flat raw * 2.5 multiplier which inflated weak matches
    (e.g. 0.15 cosine similarity → 0.375 score, appearing meaningful).
    Replaced with a tiered curve that heavily penalises weak matches and only
    amplifies genuinely strong signal.
    """
    if raw < 0.12:
        return 0.0
    if raw < 0.25:
        return round(raw * 1.2, 3)   # weak signal — minimal amplification
    if raw < 0.40:
        return round(raw * 1.8, 3)   # moderate signal
    return round(min(raw * 2.2, 1.0), 3)  # strong signal


# Domain-keyword pairs for cross-domain false-positive blocking.
# FIX: The original had no domain-awareness, causing "SWIFT" (financial protocol)
# to match "Swift" (Apple programming language) via keyword collision.
_CTI_DOMAIN_BLOCKLIST = [
    # (finding_must_contain, cti_must_NOT_contain) pairs
    ({"swift", "swiftin", "swift messaging", "swift alliance"},
     {"macos", "mac os", "apple", "xcode", "clickfix", "swift language", "swift malware", "swift stealer"}),
    ({"kubernetes", "openshift", "container", "pod"},
     {"android", "ios", "mobile", "apk", "play store"}),
    ({"android", "apk", "mobile banking"},
     {"kubernetes", "openshift", "container"}),
]


def map_cti(df, cti):
    cti = cti.copy()
    cols = {c.lower().strip(): c for c in cti.columns}
    def gc(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    inc_col = gc("incident name", "incident", "name", "title")
    desc_col = gc("description", "summary", "details")
    src_col = gc("source", "reference", "url")
    tech_col = gc("technique id", "mitre id", "attack id")
    kw_col = gc("keywords", "tags")
    type_col = gc("threat type", "category", "attack type")
    for c in [inc_col, desc_col, src_col, tech_col, kw_col, type_col]:
        if c is None:
            continue
        cti[c] = cti[c].fillna("").astype(str)
    cti_text = []
    for _, r in cti.iterrows():
        parts = []
        for c in [inc_col, desc_col, tech_col, kw_col, type_col]:
            if c: parts.append(str(r.get(c,"")))
        cti_text.append(" ".join(parts).lower())
    finding_text = [text_of(r) + " " + str(r.get("mitre_primary_id", "")) + " " + str(r.get("control_domain", "")) for _, r in df.iterrows()]
    if not cti_text or not finding_text:
        return pd.DataFrame(index=df.index)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    mat = vec.fit_transform(finding_text + cti_text)
    sim = cosine_similarity(mat[:len(finding_text)], mat[len(finding_text):])
    rows=[]
    for i in range(len(df)):
        j = int(np.argmax(sim[i]))
        score = float(sim[i,j])
        r = cti.iloc[j]
        # FIX: Raised threshold from 0.08 → 0.12 to reduce noise matches
        if score < 0.12:
            rows.append({"cti_match_score":0.0,"matched_incident":"No CTI incident match","cti_business_meaning":"No relevant historical incident was found in the uploaded CTI dataset.","cti_source":"","cti_match_terms":""})
            continue
        # FIX: Domain-aware false-positive block — prevents cross-domain keyword collisions
        # (e.g. SWIFT financial protocol matching Swift programming language CTI articles)
        finding_lower = finding_text[i].lower()
        cti_lower = cti_text[j].lower()
        blocked = False
        for finding_terms, cti_deny_terms in _CTI_DOMAIN_BLOCKLIST:
            if any(ft in finding_lower for ft in finding_terms):
                if any(dt in cti_lower for dt in cti_deny_terms):
                    blocked = True
                    break
        if blocked:
            rows.append({"cti_match_score":0.0,"matched_incident":"No CTI incident match","cti_business_meaning":"No relevant historical incident was found in the uploaded CTI dataset.","cti_source":"","cti_match_terms":""})
            continue
        # derive terms from intersection of non-stop words
        ftoks = set(re.findall(r"[a-zA-Z0-9_.-]{3,}", finding_lower))
        ctoks = set(re.findall(r"[a-zA-Z0-9_.-]{3,}", cti_lower))
        common = sorted(list(ftoks & ctoks))[:12]
        incident = str(r.get(inc_col, "Matched CTI incident")) if inc_col else "Matched CTI incident"
        src = str(r.get(src_col, "")) if src_col else ""
        rows.append({
            # FIX: Non-linear scaling replaces flat *2.5 — see _scale_cti_score()
            "cti_match_score": _scale_cti_score(score),
            "matched_incident": incident,
            "cti_business_meaning": f"This finding resembles historical threat/incident patterns in '{incident}', increasing external threat relevance.",
            "cti_source": src,
            "cti_match_terms": ", ".join(common)
        })
    return pd.DataFrame(rows, index=df.index)


def cluster_findings(df, k=None):
    n = len(df)
    if n < 5:
        df["ai_cluster"] = "Cluster 1"
        return df
    texts = [text_of(r) for _, r in df.iterrows()]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(texts)
    if k is None:
        k = min(8, max(2, int(math.sqrt(n / 2))))
    k = min(k, n)  # FIX: prevent KMeans crash when k > number of samples
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    names = []
    for c in range(k):
        center = km.cluster_centers_[c]
        top = terms[center.argsort()[-3:][::-1]]
        names.append(" / ".join(top))
    df = df.copy()
    df["ai_cluster"] = [f"Cluster {l+1}: {names[l]}" for l in labels]
    return df


def management_summary(df):
    open_df = df[~df["is_closed"]]
    overall = round(open_df["risk_score"].mean() if len(open_df) else df["risk_score"].mean(),1)
    breach = round(open_df["breach_probability"].mean() if len(open_df) else df["breach_probability"].mean(),1)
    critical = int((open_df["risk_gauge"].astype(str)=="Critical Exposure").sum())
    high = int((open_df["risk_gauge"].astype(str)=="High Exposure").sum())
    top_control = open_df.groupby("control_domain")["risk_score"].mean().sort_values(ascending=False).head(1)
    top_control_name = top_control.index[0] if len(top_control) else "N/A"
    return pd.DataFrame([
        {"Management Question":"Where do we stand now?", "Answer":f"Current open-risk posture is {risk_label(overall)} with average risk score {overall}/100 and estimated breach probability {breach}%."},
        {"Management Question":"What requires attention?", "Answer":f"There are {critical} critical-exposure and {high} high-exposure open findings. Highest pressure control domain: {top_control_name}."},
        {"Management Question":"What may happen in the future?", "Answer":future_summary(df)},
        {"Management Question":"How was MITRE mapped?", "Answer":"The platform uses control classification, allowed/blocked ATT&CK constraints, and confidence thresholds. Weak matches are marked as No reliable mapping instead of being forced."},
        {"Management Question":"How was CTI used?", "Answer":"Uploaded CTI/past incident datasets are matched to findings using text similarity plus MITRE/control context; matches show incident name, confidence, common terms, and business meaning."},
    ])


def risk_label(score):
    if score >= 80: return "Critical Exposure"
    if score >= 60: return "High Exposure"
    if score >= 30: return "Elevated"
    return "Stable"


def future_summary(df):
    open_df = df[~df["is_closed"]]
    if len(open_df)==0: return "No open findings available for forecast."
    avg_age = open_df["age_days"].mean()
    overdue_rate = open_df["sla_breached"].mean()
    now = open_df["risk_score"].mean()
    day30 = min(100, now + overdue_rate*8 + avg_age/365*3)
    day60 = min(100, now + overdue_rate*14 + avg_age/365*5)
    day90 = min(100, now + overdue_rate*20 + avg_age/365*8)
    return f"If remediation velocity does not improve, posture may move from {risk_label(now)} now to {risk_label(day90)} within 90 days. Forecast scores: 30d={day30:.1f}, 60d={day60:.1f}, 90d={day90:.1f}."


def forecast_frame(df):
    open_df = df[~df["is_closed"]]
    now = open_df["risk_score"].mean() if len(open_df) else df["risk_score"].mean()
    overdue_rate = open_df["sla_breached"].mean() if len(open_df) else 0
    avg_age = open_df["age_days"].mean() if len(open_df) else 0
    vals=[]
    for d in [0,30,60,90]:
        score = min(100, now + overdue_rate*(d/90)*20 + avg_age/365*(d/90)*8)
        vals.append({"Period":f"Today" if d==0 else f"{d} Days", "Forecast Risk Score":round(score,1), "Gauge":risk_label(score)})
    return pd.DataFrame(vals)


def to_excel(df, summary, forecast):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, index=False, sheet_name="Management Summary")
        df.to_excel(writer, index=False, sheet_name="Enriched Risk Register")
        cols = ["Finding ID","Title","control_domain","sub_control","mitre_primary_id","mitre_primary_name","mitre_tactic","mitre_confidence_label","mitre_confidence","mitre_metadata_source","mitre_reason","mitre_matching_terms","mitre_secondary","mitre_platforms","mitre_url","mitre_detection"]
        df[[c for c in cols if c in df.columns]].to_excel(writer, index=False, sheet_name="MITRE Mapping")
        ccols = ["Finding ID","Title","matched_incident","cti_match_score","cti_match_terms","cti_business_meaning","cti_source"]
        df[[c for c in ccols if c in df.columns]].to_excel(writer, index=False, sheet_name="CTI Incident Mapping")
        forecast.to_excel(writer, index=False, sheet_name="Forecast")
        df.groupby(["control_domain","risk_gauge"], observed=False).size().reset_index(name="Findings").to_excel(writer, index=False, sheet_name="Control Heatmap Data")
    return output.getvalue()


def make_sample_data():
    rows = [
        ["CSBD-001","Missing MFA enforcement for VPN users","VPN access does not enforce MFA for privileged users","High","Open","ICT Network","Digital Channels","Remote Access","Fortinet VPN","Weak authentication","Remote Access Security","2026-01-10","2026-03-01","","Production","Critical","Yes"],
        ["CSBD-002","Broad firewall rule exposure","Firewall rules allow Any-Any traffic between environments","High","Open","ICT Network","Retail Banking","Network","Firewall","Lack of segmentation","Network Security","2026-02-01","2026-04-01","","Production","High","Internal"],
        ["CSBD-003","Lack of service-to-service encryption","Internal microservice traffic is not protected with mTLS","Medium","In Progress","Digital Banking","Digital Channels","OpenShift","Kubernetes","Missing encryption","Cryptography","2026-02-20","2026-05-01","","Production","High","Internal"],
        ["CSBD-004","Insufficient logging retention","Security logs are retained for less than policy requirement","Medium","Open","SOC","Risk Management","Monitoring","SIEM","Retention gap","Detection","2026-03-10","2026-04-20","","Production","Medium","Internal"],
        ["CSBD-005","Public cloud storage exposure","Sensitive files stored in publicly accessible cloud bucket","Critical","Open","Cloud Team","Corporate Banking","Cloud","AWS S3","Misconfiguration","Cloud Security","2026-04-01","2026-04-15","","Production","Critical","Yes"],
        ["CSBD-006","Weak API authentication","APIs rely on static tokens without rotation","High","Open","Digital Banking","Payments","Application","API Gateway","Poor token governance","API Security","2026-04-05","2026-05-15","","Production","Critical","Yes"],
        ["CSBD-007","Missing Kubernetes Network Policies","Pods communicate without namespace segmentation restrictions","High","Open","Cloud Team","Payments","OpenShift","Kubernetes","Lack of segmentation","Container Security","2026-04-08","2026-05-10","","Production","Critical","Internal"],
        ["CSBD-008","Missing CIS hardening","Servers do not comply with CIS benchmark","Medium","Open","Systems Administration","Treasury","Infrastructure","Windows Server","Missing hardening","Hardening","2026-03-01","2026-06-01","","Production","High","Internal"],
        ["CSBD-009","Excessive Active Directory privileges","Users assigned Domain Admin unnecessarily","Critical","Open","Systems Administration","Corporate Banking","IAM","Active Directory","Poor access control","Identity","2026-02-15","2026-03-15","","Production","Critical","Internal"],
        ["CSBD-010","Shared administrative accounts","Multiple administrators use same local admin account","High","Risk Accepted","Systems Administration","Retail Banking","PAM","Windows Server","Weak governance","Privileged Access","2026-01-20","2026-03-20","","Production","High","Internal"],
    ]
    cols=["Finding ID","Title","Description","Severity","Status","Owner","Business Unit","Domain","Technology","Root Cause","Risk Category","Created Date","Due Date","Closure Date","Environment","Asset Criticality","Internet Facing"]
    return pd.DataFrame(rows, columns=cols)


def make_sample_cti():
    rows = [
        ["MOVEit-style Data Theft","Attackers exploited exposed applications and stole data from enterprise systems.","Data Theft","Financial Services","T1190","Exploit Public-Facing Application","Public reporting","2023","public-facing, exploit, data theft, application"],
        ["Cloud Bucket Exposure Incident","Sensitive customer files were collected from misconfigured public cloud storage.","Cloud Data Exposure","Financial Services","T1530","Data from Cloud Storage","Public reporting","2022","bucket, cloud storage, public, sensitive data"],
        ["VPN Account Compromise Campaign","Threat actors used exposed VPN access and weak authentication to gain initial access.","Ransomware Initial Access","Banking","T1133","External Remote Services","Public reporting","2024","vpn, mfa, remote access, initial access"],
        ["API Token Abuse Case","Stolen application access tokens were used to access backend APIs.","API Abuse","Technology","T1528","Steal Application Access Token","Public reporting","2024","api, token, jwt, oauth, static token"],
        ["Ransomware Lateral Movement","Adversaries moved laterally through reachable internal remote services after initial compromise.","Ransomware","Financial Services","T1021","Remote Services","Public reporting","2023","firewall, segmentation, remote services, lateral movement"],
        ["Credential Spraying Against Bank","Weak password patterns enabled large-scale password spraying against user accounts.","Credential Attack","Banking","T1110.003","Password Spraying","Public reporting","2021","password, weak password, password policy, spraying"],
        ["Log Deletion and Monitoring Bypass","Attackers disabled monitoring tools and reduced forensic visibility.","Defense Evasion","Enterprise","T1562","Impair Defenses","Public reporting","2022","logging, siem, retention, monitoring, detection"],
        ["Unencrypted Internal Traffic Capture","Attackers captured sensitive internal traffic where encryption in transit was missing.","Credential/Data Interception","Enterprise","T1040","Network Sniffing","Public reporting","2020","unencrypted, tls, mtls, service-to-service, sniffing"],
    ]
    return pd.DataFrame(rows, columns=["Incident Name","Description","Threat Type","Industry","Technique ID","Technique Name","Source","Year","Keywords"])


def main():
    st.title("Cyber Assurance Predictive Intelligence Platform")
    st.caption("Hybrid ML/statistical/rule-based engine. No LLM. Designed for management reporting and cybersecurity-by-design risk analysis.")

    with st.sidebar:
        st.header("Inputs")
        risk_file = st.file_uploader("Upload Risk Register CSV/XLSX", type=["csv","xlsx"])
        cti_file = st.file_uploader("Optional: Upload CTI / Kaggle Past Incidents CSV/XLSX", type=["csv","xlsx"])
        mitre_file = st.file_uploader("Optional: Upload MITRE ATT&CK STIX JSON", type=["json"], help="Upload enterprise-attack.json from the official mitre-attack/attack-stix-data repository.")
        use_sample = st.checkbox("Use sample risk register", value=(risk_file is None))
        use_sample_cti = st.checkbox("Use sample CTI/past incidents", value=(cti_file is None))
        use_sample_mitre = st.checkbox("Use sample MITRE metadata", value=(mitre_file is None))
        st.divider()
        st.write("Confidence policy")
        st.caption("MITRE mappings are constrained. Uploaded STIX enriches official technique names/tactics; weak cases become 'No reliable mapping'.")

    if risk_file is not None:
        raw = load_tabular(risk_file)
    elif use_sample:
        raw = make_sample_data()
    else:
        st.info("Upload a risk register or enable sample data.")
        return

    if cti_file is not None:
        cti = load_tabular(cti_file)
    elif use_sample_cti:
        cti = make_sample_cti()
    else:
        cti = None

    mitre_lookup = {}
    mitre_df = pd.DataFrame()
    if mitre_file is not None:
        mitre_lookup, mitre_df = parse_mitre_stix(load_json_file(mitre_file))
    elif use_sample_mitre:
        sample_mitre_path = "data/sample_enterprise_attack_min.json"
        try:
            mitre_lookup, mitre_df = parse_mitre_stix(load_json_file(sample_mitre_path))
        except Exception:
            mitre_lookup, mitre_df = {}, pd.DataFrame()

    df = normalize_cols(raw)
    df = enrich_scores(df, cti, mitre_lookup)
    df = cluster_findings(df)
    summary = management_summary(df)
    forecast = forecast_frame(df)

    open_df = df[~df["is_closed"]]
    avg_risk = round(open_df["risk_score"].mean() if len(open_df) else df["risk_score"].mean(),1)
    avg_breach = round(open_df["breach_probability"].mean() if len(open_df) else df["breach_probability"].mean(),1)
    critical_count = int((open_df["risk_gauge"].astype(str)=="Critical Exposure").sum())
    high_count = int((open_df["risk_gauge"].astype(str)=="High Exposure").sum())

    st.subheader("Management Snapshot")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk Score", f"{avg_risk}/100", risk_label(avg_risk))
    c2.metric("Breach Probability", f"{avg_breach}%")
    c3.metric("Critical Exposure", critical_count)
    c4.metric("High Exposure", high_count)

    gauge = go.Figure(go.Indicator(mode="gauge+number", value=avg_risk, title={"text":"Overall Risk Gauge"}, gauge={"axis":{"range":[0,100]}, "steps":[{"range":[0,30]}, {"range":[30,60]}, {"range":[60,80]}, {"range":[80,100]}]}))
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("Where We Stand / What May Happen")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.line_chart(forecast.set_index("Period")[["Forecast Risk Score"]])

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Management View","Control Taxonomy","MITRE Mapping","CTI Incident Mapping","Analytics","Export"])

    with tab1:
        cols = ["Finding ID","Title","Severity","Status","Owner","Business Unit","risk_score","breach_probability","risk_gauge","threat_classification","where_we_stand","future_outlook","management_action"]
        st.dataframe(df[cols].sort_values("risk_score", ascending=False), use_container_width=True, hide_index=True)

    with tab2:
        cols = ["Finding ID","Title","control_domain","sub_control","control_classification_confidence","control_criticality_multiplier","threat_classification","risk_score"]
        st.dataframe(df[cols].sort_values("risk_score", ascending=False), use_container_width=True, hide_index=True)
        heat = df.groupby(["control_domain","risk_gauge"], observed=False).size().reset_index(name="Findings")
        st.plotly_chart(px.bar(heat, x="control_domain", y="Findings", color="risk_gauge", title="Control Domains by Risk Gauge"), use_container_width=True)

    with tab3:
        cols = ["Finding ID","Title","control_domain","mitre_primary_id","mitre_primary_name","mitre_tactic","mitre_confidence_label","mitre_confidence","mitre_metadata_source","mitre_reason","mitre_matching_terms","mitre_secondary","mitre_platforms","mitre_url"]
        display_cols = [c for c in cols if c in df.columns]
        st.dataframe(df.sort_values(["mitre_confidence","risk_score"], ascending=False)[display_cols], use_container_width=True, hide_index=True)
        mitre_counts = df[df["mitre_primary_id"]!="No reliable mapping"].groupby(["mitre_primary_id","mitre_primary_name"]).size().reset_index(name="Findings").sort_values("Findings", ascending=False)
        st.plotly_chart(px.bar(mitre_counts, x="mitre_primary_id", y="Findings", hover_data=["mitre_primary_name"], title="Mapped MITRE Techniques"), use_container_width=True)
        if mitre_df is not None and len(mitre_df):
            st.subheader("Uploaded MITRE ATT&CK techniques loaded")
            st.caption(f"{len(mitre_df):,} active techniques/sub-techniques loaded from STIX. Mapping remains constrained by control-domain rules.")
            st.dataframe(mitre_df[["mitre_id","mitre_name","mitre_tactics","mitre_platforms","mitre_url"]].head(300), use_container_width=True, hide_index=True)

    with tab4:
        cols = ["Finding ID","Title","matched_incident","cti_match_score","cti_match_terms","cti_business_meaning","cti_source","risk_score"]
        st.dataframe(df.sort_values(["cti_match_score","risk_score"], ascending=False)[cols], use_container_width=True, hide_index=True)

    with tab5:
        a,b = st.columns(2)
        with a:
            st.plotly_chart(px.histogram(df, x="risk_score", nbins=20, title="Risk Score Distribution"), use_container_width=True)
            st.plotly_chart(px.bar(df.groupby("Owner")["risk_score"].mean().sort_values(ascending=False).reset_index(), x="Owner", y="risk_score", title="Average Risk by Owner"), use_container_width=True)
        with b:
            st.plotly_chart(px.pie(df, names="threat_classification", title="Threat Classification Distribution"), use_container_width=True)
            st.plotly_chart(px.bar(df.groupby("ai_cluster").size().reset_index(name="Findings"), x="ai_cluster", y="Findings", title="NLP Clusters"), use_container_width=True)

        st.subheader("Systemic Risk Indicators")
        sys = df.groupby(["control_domain","Root Cause"]).agg(Findings=("Finding ID","count"), Avg_Risk=("risk_score","mean"), SLA_Breaches=("sla_breached","sum")).reset_index().sort_values(["Avg_Risk","Findings"], ascending=False)
        st.dataframe(sys, use_container_width=True, hide_index=True)

    with tab6:
        st.download_button("Download enriched Excel report", data=to_excel(df, summary, forecast), file_name="cyber_assurance_predictive_hybrid_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download enriched CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="cyber_assurance_predictive_hybrid_output.csv", mime="text/csv")

if __name__ == "__main__":
    main()