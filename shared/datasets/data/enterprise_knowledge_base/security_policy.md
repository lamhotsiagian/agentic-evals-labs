# Information Security & Access Management Standards (DOC-SEC-202)
**Department**: Information Security  
**Effective Date**: January 1, 2026  
**Document ID**: DOC-SEC-202  

## 1. Password Complexity Requirements
* All user passwords must be a minimum of **16 characters in length**.
* Passwords must contain a combination of uppercase letters, lowercase letters, numbers, and special symbols.
* Passwords must not contain personal identifiers, sequential dictionary words, or previously compromised credentials.

## 2. Multi-Factor Authentication (MFA)
* Multi-Factor Authentication (MFA) utilizing **FIDO2 hardware security keys (e.g. YubiKey)** or time-based one-time password (TOTP) authenticator applications is mandatory for all employee accounts.
* SMS-based 2FA is explicitly prohibited for production infrastructure access.
* Sharing API secrets, SSH keys, or access tokens in team chat channels (e.g. Slack/Teams) constitutes a critical security policy violation.
