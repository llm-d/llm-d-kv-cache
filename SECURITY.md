# Security Policy

## Reporting a Vulnerability

The llm-d project takes security seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

If you discover a security vulnerability in llm-d-kv-cache, please report it by emailing **security@llm-d.io**. Please do not report security vulnerabilities through public GitHub issues.

### What to Include

To help us better understand the nature and scope of the vulnerability, please include the following information in your report:

* **Description:** A clear and concise description of the vulnerability.
* **Steps to Reproduce:** Detailed steps to reproduce the issue, including any relevant configuration or code snippets.
* **Impact:** An explanation of the potential impact of the vulnerability.
* **Affected Versions:** The version(s) of llm-d-kv-cache affected by the vulnerability.
* **Suggested Fix (Optional):** If you have ideas on how to fix the vulnerability, please include them.

### Response Timeline

We will acknowledge receipt of your vulnerability report within **3 business days** and will send a more detailed response within **7 business days** indicating the next steps in handling your report.

After the initial reply to your report, we will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

### Disclosure Policy

We follow a coordinated disclosure process:

1. We will work with you to understand and validate the reported vulnerability.
2. We will develop and test a fix.
3. We will prepare a security advisory.
4. We will release the fix and publish the security advisory.
5. We will credit you (unless you prefer to remain anonymous) in the security advisory.

We ask that you:

* Give us reasonable time to investigate and fix the vulnerability before public disclosure.
* Make a good faith effort to avoid privacy violations, destruction of data, and interruption or degradation of our services.
* Do not exploit the vulnerability beyond what is necessary to demonstrate it.

## Supported Versions

Security updates will be provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

Once stable releases are available, this policy will be updated to reflect which versions receive security updates.

## Security Best Practices

When deploying llm-d-kv-cache:

* Keep your deployment up to date with the latest stable release.
* Follow the principle of least privilege when configuring access controls.
* Monitor the project's security advisories at [https://github.com/llm-d/llm-d-kv-cache/security/advisories](https://github.com/llm-d/llm-d-kv-cache/security/advisories).
* Review and follow any security-related configuration recommendations in the documentation.

## Thank You

We appreciate the security research community's efforts in helping keep llm-d-kv-cache and our users safe. Responsible disclosure of security vulnerabilities helps us ensure the security and privacy of all users.
