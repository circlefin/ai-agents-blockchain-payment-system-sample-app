const {
  initiateDeveloperControlledWalletsClient,
} = require("@circle-fin/developer-controlled-wallets");
require("dotenv").config();

const client = initiateDeveloperControlledWalletsClient({
  apiKey: process.env.CIRCLE_API_KEY,
  entitySecret: process.env.CIRCLE_ENTITY_SECRET,
});

/**
 * Creates a new WalletSet.
 * @param name - The name of the WalletSet.
 * @returns The created WalletSet.
 * @throws Will throw an error if WalletSet creation fails.
 */
const createWalletSet = async (name) => {
  console.log("Creating WalletSet...");
  const response = await client.createWalletSet({ name });

  if (!response.data?.walletSet) {
    throw new Error("Failed to create WalletSet.");
  }

  console.log(
    `WalletSet "${name}" created with ID: ${response.data.walletSet.id}`
  );
  return response.data.walletSet;
};

/**
 * Creates wallets within the specified WalletSet.
 * @param walletSetId - The ID of the WalletSet.
 * @param blockchains - Array of blockchain identifiers.
 * @param count - Number of wallets to create.
 * @returns Array of created wallets.
 * @throws Will throw an error if wallet creation fails.
 */
const createWallets = async (walletSetId, blockchains, count) => {
  console.log("Creating wallets...");
  const response = await client.createWallets({
    blockchains,
    count,
    walletSetId,
  });

  if (!response.data?.wallets || response.data.wallets.length === 0) {
    throw new Error("Failed to create wallets.");
  }

  const addresses = response.data.wallets
    .map((wallet) => wallet.address)
    .join(", ");
  console.log(`Successfully created wallets: ${addresses}`);
  return response.data.wallets;
};

/**
 * Requests testnet tokens for a specific wallet.
 * @param address - The wallet address to request tokens for.
 * @param blockchain - The blockchain identifier.
 * @param usdc - Boolean indicating if USDC tokens should be requested.
 */
const requestTestnetTokens = async (address, blockchain, usdc) => {
  console.log(
    `Requesting testnet tokens for address: ${address} on ${blockchain} (USDC: ${usdc})...`
  );
  await client.requestTestnetTokens({
    address,
    blockchain,
    usdc,
  });
  console.log("Testnet tokens requested successfully.");
};

/**
 * Checks the balance of a wallet.
 * @param walletId - The ID of the wallet to check.
 * @returns The wallet's token balance.
 * @throws Will throw an error if balance check fails.
 */
const checkBalance = async (walletId) => {
  console.log(`Checking balance for wallet ${walletId}...`);
  try {
    const response = await client.getWalletTokenBalance({ id: walletId });
    const walletTokenBalance = response?.data;
    console.log("Wallet balance:", JSON.stringify(walletTokenBalance, null, 2));
    return walletTokenBalance;
  } catch (error) {
    console.error("Error checking wallet balance:", error);
    throw error;
  }
};

/**
 * Ensures that the wallet has a sufficient USDC balance.
 * @param walletId - The ID of the wallet to check.
 * @param requiredBalance - The minimum required balance.
 * @param maxAttempts - Maximum number of balance check attempts.
 * @param interval - Interval between balance checks in milliseconds.
 * @returns The USDC token ID if sufficient balance is found.
 * @throws Will throw an error if sufficient balance is not found within the attempts.
 */
const ensureSufficientBalance = async (walletId, requiredBalance, maxAttempts = 20, interval = 15000) => {
  console.log(`Ensuring sufficient balance for wallet ${walletId}...`);
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const balance = await checkBalance(walletId);
    const usdcBalance = balance.tokenBalances.find(token => token.token.symbol === 'USDC');
    
    if (usdcBalance && parseFloat(usdcBalance.amount) >= requiredBalance) {
      console.log(`Sufficient USDC balance found: ${usdcBalance.amount}`);
      return usdcBalance.token.id;
    }
    
    console.log(`Attempt ${attempt}: Insufficient balance. Waiting before next check...`);
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  throw new Error(`Failed to find sufficient USDC balance after ${maxAttempts} attempts.`);
};

/**
 * Creates a transaction between two wallets.
 * @param amount - Array containing the amount to transfer.
 * @param walletId - The ID of the source wallet.
 * @param destinationAddress - The address of the destination wallet.
 * @param tokenId - The ID of the token to transfer.
 * @returns The created transaction.
 * @throws Will throw an error if transaction creation fails.
 */
const createTransaction = async (
  amount,
  walletId,
  destinationAddress,
  tokenId
) => {
  console.log("Creating transaction...");
  const transactionResponse = await client.createTransaction({
    amounts: [amount],
    walletId,
    destinationAddress,
    tokenId,
    fee: {
      type: "level",
      config: {
        feeLevel: "HIGH",
      },
    },
  });

  if (!transactionResponse.data) {
    throw new Error("Failed to create transaction.");
  }

  console.log(`Transaction initiated with ID: ${transactionResponse.data.id}`);
  return transactionResponse.data;
};

/**
 * Waits until the specified transaction reaches the 'CONFIRMED' state.
 * @param transactionId - The ID of the transaction to monitor.
 * @returns The confirmed transaction object.
 * @throws Will throw an error if the transaction is not confirmed within the timeout.
 */
const waitForTransactionConfirmation = async (
  transactionId,
  timeout = 240000,
  interval = 200
) => {
  console.log(`Waiting for transaction ${transactionId} to be confirmed...`);
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const transactionResponse = await client.getTransaction({
      id: transactionId,
    });

    const state = transactionResponse.data?.transaction?.state;
    if (state) {
      console.log(`Current transaction state: ${state}`);

      if (state === "CONFIRMED") {
        console.log("Transaction confirmed successfully.");
        return transactionResponse.data.transaction;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error(
    "Timeout: Transaction was not confirmed in the expected time."
  );
};

/**
 * Validates that the payment was received by the agent's wallet.
 * @param walletId - The ID of the agent's wallet.
 * @param expectedAmount - The expected payment amount.
 * @returns Boolean indicating if the payment was received.
 */
const validatePayment = async (walletId, expectedAmount) => {
  const balance = await checkBalance(walletId);
  const usdcBalance = balance.tokenBalances.find(token => token.token.symbol === 'USDC');
  
  if (usdcBalance && parseFloat(usdcBalance.amount) >= parseFloat(expectedAmount)) {
    console.log(`Payment validated for wallet ${walletId}. Expected: ${expectedAmount}, Actual: ${usdcBalance.amount}`);
    return true;
  } else {
    console.log(`Payment validation failed for wallet ${walletId}. Expected: ${expectedAmount}, Actual: ${usdcBalance?.amount || '0'}`);
    return false;
  }
};

/**
 * Main function to orchestrate wallet creation, token request, transaction processing for multiple agents.
 */
const processPayments = async (contributions) => {
  const results = { success: true, payments: {} };
  const paymentSummary = [];

  try {
    const walletSet = await createWalletSet("AI Agents Wallet Set");
    const mainWallet = (await createWallets(walletSet.id, ["SOL-DEVNET"], 1))[0];
    const agentWallets = await createWallets(walletSet.id, ["SOL-DEVNET"], Object.keys(contributions).length);

    await requestTestnetTokens(mainWallet.address, "SOL-DEVNET", true);

    const requiredBalance = Object.values(contributions).reduce((sum, value) => sum + value, 0);
    const tokenId = await ensureSufficientBalance(mainWallet.id, requiredBalance);

    for (const [agent, contribution] of Object.entries(contributions)) {
      const agentWallet = agentWallets[Object.keys(contributions).indexOf(agent)];
      const transaction = await createTransaction(
        contribution.toString(),
        mainWallet.id,
        agentWallet.address,
        tokenId
      );

      await waitForTransactionConfirmation(transaction.id);

      const balance = await checkBalance(agentWallet.id);
      const usdcBalance = balance.tokenBalances.find(token => token.token.symbol === 'USDC');

      paymentSummary.push({
        agent,
        balance: usdcBalance ? usdcBalance.amount : '0',
        address: agentWallet.address,
        blockchain: 'SOL-DEVNET'
      });
    }

    console.log("\n=== Payment Summary ===");
    console.table(paymentSummary);

  } catch (error) {
    results.success = false;
    results.error = error.message;
  }

  return results;
};

// Read contributions from stdin and process payments
let data = "";
process.stdin.on("data", (chunk) => {
  data += chunk;
});
process.stdin.on("end", async () => {
  const contributions = JSON.parse(data);
  const paymentResults = await processPayments(contributions);
  console.log(JSON.stringify(paymentResults));
});