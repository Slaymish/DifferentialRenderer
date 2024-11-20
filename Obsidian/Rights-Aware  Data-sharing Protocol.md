# Rights-Aware Data-sharing Protocol and Client Application

- Cryptographically enforced
- Protocol designed to embed legally binding, user-defined rights into the data itself, ensuring compliance even after the data is shared or downloaded

## Components

1. Consent Management
2. Persistent rights metadata
3. Decentralised verification
4. Peer-to-peer communication
5. Accountability and traceability
6. Legal binding


## Product

1. Website

- Acts as hub for agreement creation and cryptographic proof management
- Provides users with tools to:
	- Define terms and conditions
	- Generate and manage cryptographic keys
	- Initiate connects with other users

2. Client application

- Packages data with embedded rights metadata
- Encrypts and send data using the protocol
- Allows recipients to view and verify consent details
- Only only users to view the data someone send once they have agreed to the same terms and conditions to make it legally binding


3. Blockchain integration

- Records agreements and key transactions, ensuring decentralised verification
- Provides tools for auditing data usage and proving rights violations

## Example

1. Alice uploads image to website and specifies:

- Terms: "Non-commercial use only, no third-party sharing"
- Consent is hashed, signed, and stored on blockchain

2. Alice shares image with Bob via client

- Image includes metadata with consent terms, cryptographic proof, and a hash for integrity verification

3. Bob downloads image and tries to share with third-party

- Third party client verifies the metadata and denies access because the term prohibit third-party sharing

4. If Bob bypasses the system, Alice has cryptographic proof to demonstrate a violation


